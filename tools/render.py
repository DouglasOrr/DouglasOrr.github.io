import argparse
import glob
import html
import io
import json
import logging
import os
import re
import shutil
import sys
import urllib.request
from pathlib import Path

import inotify.adapters
import jsmin
import markdown


class RenameLinksProcessor(markdown.treeprocessors.Treeprocessor):
    """Rename markdown links to HTML ones."""

    def run(self, root):
        for link in root.iter("a"):
            href = link.attrib["href"]
            if href.endswith(".md"):
                link.attrib["href"] = re.sub(".md$", ".html", href)


class PrismFencedBlockPreprocessor(markdown.preprocessors.Preprocessor):
    LANGUAGE_NAME_MAP = {"c++": "cpp", "json": "javascript"}
    BLOCK = re.compile(
        r"```(?P<language>[^\n]*)\n(?P<code>.+?)\n```", flags=re.MULTILINE | re.DOTALL
    )

    def _replace(self, m):
        language = m.group("language")
        clazz = "language-" + (self.LANGUAGE_NAME_MAP.get(language, language) or "none")
        code = html.escape(m.group("code"))
        return self.md.htmlStash.store(
            f'<pre><code class="{clazz}">{code}</code></pre>'
        )

    def run(self, lines):
        return self.BLOCK.sub(self._replace, "\n".join(lines)).split("\n")


class BootstrapProcessor(markdown.treeprocessors.Treeprocessor):
    """Set bootstrap styling classes."""

    def run(self, root):
        for element in root.iter("table"):
            element.attrib["class"] = "table"
        for element in root.iter("img"):
            element.attrib["class"] = "img-fluid"
        for element in root.iter("blockquote"):
            element.attrib["class"] = "blockquote"


class DougsDiversionsExtension(markdown.extensions.Extension):
    """Custom extension for this site."""

    def extendMarkdown(self, md):
        md.preprocessors.register(
            PrismFencedBlockPreprocessor(md), "prism_fenced_code", 26
        )
        md.treeprocessors.register(RenameLinksProcessor(md), "rename_links", 19)
        md.treeprocessors.register(BootstrapProcessor(md), "custom_style", 18)


class Rule:
    """A rule can be asked to build or delete a single target file in the output."""

    def __init__(self, target):
        self.target = target

    def delete_target(self):
        if os.path.isfile(self.target):
            logging.info(f"delete {self.target}")
            os.remove(self.target)
        else:
            logging.warning(f"cannot delete {self.target} - not a file")

    def build_target(self):
        os.makedirs(os.path.dirname(self.target), exist_ok=True)
        self._build()

    def _build(self):
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, Rule) and other.target == self.target

    def __hash__(self):
        return hash(self.target)

    def __lt__(self, other):
        return self.target < other.target


class CopyRule(Rule):
    def __init__(self, target, source):
        super().__init__(target)
        self.source = source

    def _build(self):
        logging.info(f"copy {self.source} => {self.target}")
        shutil.copyfile(self.source, self.target)


class RenderRule(Rule):
    MARKDOWN = markdown.Markdown(
        extensions=[
            DougsDiversionsExtension(),
            "meta",
            "tables",
            "toc",
            "footnotes",
            "md_in_html",
        ]
    )

    def __init__(self, target, source, template, root):
        super().__init__(target)
        self.source = source
        self.template = template
        self.root = root

    @classmethod
    def _render_markdown(cls, source: str) -> tuple[str, dict[str, list[str]]]:
        cls.MARKDOWN.reset()
        html = cls.MARKDOWN.convert(source)
        return html, cls.MARKDOWN.Meta

    @classmethod
    def _render_notebook(cls, source: str) -> tuple[str, dict[str, list[str]]]:
        html = []
        meta = {}
        notebook = json.loads(source)
        if (nbformat := notebook["nbformat"]) != 4:
            print(
                "Warning - this renderer was tested on nbformat:4,"
                f" this notebook has: nbformat:{nbformat}",
                file=sys.stderr,
            )
        for cell in notebook["cells"]:
            if cell["cell_type"] == "markdown":
                cls.MARKDOWN.reset()
                html.append(cls.MARKDOWN.convert("".join(cell["source"])))
                for key, value in cls.MARKDOWN.Meta.items():
                    if key in meta:
                        print(f"Warning - duplicate meta key: {key}", file=sys.stderr)
                meta.update(cls.MARKDOWN.Meta)
            for output in cell.get("outputs", []):
                if output["output_type"] == "stream":
                    if output["name"] != "stderr":
                        html.append(f"<pre>{''.join(output['text'])}</pre>")
                elif output["output_type"] == "display_data":
                    if "image/png" in output["data"]:
                        html.append(
                            f'<img src="data:image/png;base64,{output["data"]["image/png"]}"/>'
                        )
                    else:
                        print(
                            "Warning - unhandled output_type: display_data mime types:"
                            f" {list(output['data'].keys())}",
                            file=sys.stderr,
                        )
                else:
                    print(
                        f"Warning - unhandled output_type: {output['output_type']}",
                        file=sys.stderr,
                    )
        return "".join(html), meta

    def _build(self):
        self.MARKDOWN.reset()
        logging.info(f"render {self.source} => {self.target}")
        with open(self.template, encoding="utf-8") as template_f:
            template = template_f.read()
        with open(self.source, encoding="utf-8") as source_f:
            contents = source_f.read()
            if self.source.endswith(".md"):
                body, meta = self._render_markdown(contents)
            elif self.source.endswith(".ipynb"):
                body, meta = self._render_notebook(contents)
            else:
                raise ValueError(f"Unsupported source type: {self.source}")
        title = " ".join(meta["title"])
        keywords = ",".join(meta["keywords"])
        og_meta = []
        og_meta.append(f'<meta property="og:title" content="{title}">')
        if "image" in meta:
            src = (
                Path(Path(self.target).parent, meta["image"][0])
                .resolve()
                .relative_to(Path(self.root).resolve())
            )
            og_meta.append(f'<meta property="og:image" content="/{src}">')
        if "description" in self.MARKDOWN.Meta:
            description = " ".join(self.MARKDOWN.Meta["description"])
            og_meta.append(f'<meta property="og:description" content="{description}">')
        html = (
            template.replace("{{title}}", title)
            .replace("{{keywords}}", keywords)
            .replace("{{og-meta}}", "\n".join(og_meta))
            .replace("{{body}}", body)
        )
        with open(self.target, "w", encoding="utf-8") as target_f:
            target_f.write(html)


class DownloadRule(Rule):
    def __init__(self, target, sources):
        super().__init__(target)
        self.sources = sources

    def _build(self):
        logging.info(f'download {" + ".join(self.sources)} => {self.target}')
        tmpfile = io.StringIO()
        for source in self.sources:
            with urllib.request.urlopen(source) as response:
                tmpfile.write(response.read().decode("utf8"))
        # Automatically minify js
        data = tmpfile.getvalue()
        if self.target.endswith(".js"):
            data = jsmin.jsmin(data)
        with open(self.target, "w") as target_f:
            target_f.write(data)


class DownloadFontsRule(Rule):
    """Download fonts from the Google Fonts CDN (use the "embed code" link)."""

    @staticmethod
    def download_fonts(embed_url: str, out: Path) -> None:
        out.mkdir(exist_ok=True)
        user_agent = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36"
        request = urllib.request.Request(embed_url, headers={"User-Agent": user_agent})
        code = urllib.request.urlopen(request).read().decode("utf-8")
        for url in set(re.findall(r"url\((https://[^)]+)\)", code)):
            *_, font, version, hashname = url.split("/")
            name = f"{font}-{version}-{hashname}"
            urllib.request.urlretrieve(url, out / name)
            code = code.replace(url, name)
        (out / "fonts.css").write_text(code)

    def __init__(self, target, embed_url):
        super().__init__(target)
        self.embed_url = embed_url

    def _build(self):
        logging.info(f"download fonts {self.embed_url} => {self.target}")
        self.download_fonts(self.embed_url, Path(self.target))


class Builder:
    DEST_NOCLEAN = {".git", "README.md", "server.log", ".nojekyll"}
    DEST_LIBS = [
        (
            "css/lib.css",
            [
                "https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css",
                "https://cdn.jsdelivr.net/npm/prismjs@1.21.0/themes/prism.css",
            ],
        ),
        (
            "js/lib.js",
            [
                "https://cdn.jsdelivr.net/npm/mathjax@3.1.4/es5/tex-svg.js",
                "https://cdn.jsdelivr.net/npm/prismjs@1.21.0/prism.js",
            ]
            + [
                f"https://cdn.jsdelivr.net/npm/prismjs@1.21.0/components/prism-{language}.js"
                for language in [
                    "clike",
                    "javascript",
                    "c",
                    "cpp",
                    "java",
                    "python",
                    "typescript",
                ]
            ],
        ),
    ]
    FONTS = "https://fonts.googleapis.com/css2?family=Inconsolata:wght@200..900&family=Jost:ital,wght@0,100..900;1,100..900&display=swap"
    SRC_TEMPLATE = "template.html"
    SRC_IGNORE = {".ipynb_checkpoints", "ext"}
    SRC_COPY = {".html", ".png", ".jpg", ".gif", ".svg", ".mp4", ".css", ".js"}

    class Error(Exception):
        def __init__(self, path, description):
            self.path = path
            super().__init__(f'Error building "{path}" - {description}')

    def __init__(self, src_root, dest_root):
        self.src_root = src_root
        self.dest_root = dest_root

    def _check_ignore(self, path):
        parts = path.split(os.path.sep)
        if any(part in self.SRC_IGNORE for part in parts):
            return True
        if parts[-1].startswith(".#"):
            return True
        return False

    def _src_to_dest(self, src):
        return (
            os.path.join(self.dest_root, os.path.relpath(src, self.src_root))
            .replace(".md", ".html")
            .replace(".ipynb", ".html")
        )

    def _render_rule(self, src):
        return RenderRule(
            self._src_to_dest(src),
            src,
            template=os.path.join(self.src_root, self.SRC_TEMPLATE),
            root=self.dest_root,
        )

    def _get_rules(self, src):
        """Gets the set of rules that apply from this source path."""
        if self._check_ignore(src):
            logging.info(f"ignore {src}")
            return set([])
        ext = os.path.splitext(src)[1]
        if ext in self.SRC_COPY:
            return {CopyRule(self._src_to_dest(src), src)}
        if ext == ".md" or ext == ".ipynb":
            return {self._render_rule(src)}
        if os.path.relpath(src, self.src_root) == self.SRC_TEMPLATE:
            return {
                self._render_rule(src)
                for ext in [".md", ".ipynb"]
                for src in glob.glob(f"{self.src_root}/**/*{ext}", recursive=True)
            }
        raise self.Error(src, "missing build rule")

    def clean(self):
        os.makedirs(self.dest_root, exist_ok=True)
        for name in os.listdir(self.dest_root):
            if name not in self.DEST_NOCLEAN:
                path = os.path.join(self.dest_root, name)
                logging.info(f"clean {path}")
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

    def rebuild(self):
        self.clean()
        rules = set([])
        # Add rules for each file in the source tree
        for parent, _, files in os.walk(self.src_root):
            for file in files:
                rules |= self._get_rules(os.path.join(parent, file))
        # Add library rules
        for target, source in self.DEST_LIBS:
            rules |= {
                DownloadRule(
                    os.path.join(self.dest_root, target),
                    [source] if isinstance(source, str) else source,
                )
            }
        # Add font download rule
        rules.add(DownloadFontsRule(os.path.join(self.dest_root, "fonts"), self.FONTS))
        # Run all the rules
        for rule in sorted(rules):
            rule.build_target()

    def watch(self):
        events = inotify.adapters.InotifyTree(self.src_root).event_gen(
            yield_nones=False
        )
        for _, types, parent, name in events:
            try:
                if "IN_ISDIR" not in types:
                    src = os.path.join(parent, name)
                    if "IN_CLOSE_WRITE" in types:
                        for rule in self._get_rules(src):
                            rule.build_target()
                    if "IN_DELETE" in types:
                        for rule in self._get_rules(src):
                            rule.delete_target()
            except self.Error as e:
                logging.error(e)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="Site render script")
    parser.add_argument("src")
    parser.add_argument("dest")
    parser.add_argument(
        "--dev", action="store_true", help="watch for changes and re-render"
    )
    args = parser.parse_args()

    builder = Builder(args.src, args.dest)
    builder.rebuild()
    if args.dev:
        builder.watch()
