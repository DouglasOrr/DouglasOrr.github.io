import argparse
import glob
import html
import inotify.adapters
import io
import jsmin
import logging
import markdown
import os
import re
import shutil
import urllib.request


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
        ]
    )

    def __init__(self, target, source, template):
        super().__init__(target)
        self.source = source
        self.template = template

    def _build(self):
        logging.info(f"render {self.source} => {self.target}")
        with open(self.template, encoding="utf-8") as template_f:
            template = template_f.read()
        with open(self.source, encoding="utf-8") as source_f:
            body = self.MARKDOWN.convert(source_f.read())
        title = " ".join(self.MARKDOWN.Meta["title"])
        keywords = ",".join(self.MARKDOWN.Meta["keywords"])
        html = (
            template.replace("{{title}}", title)
            .replace("{{keywords}}", keywords)
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


class Builder:
    DEST_NOCLEAN = {".git", "README.md", "server.log"}
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
                "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js",
                "https://cdn.jsdelivr.net/npm/prismjs@1.21.0/prism.js",
            ]
            + [
                f"https://cdn.jsdelivr.net/npm/prismjs@1.21.0/components/prism-{language}.js"
                for language in ["clike", "javascript", "c", "cpp", "java", "python"]
            ],
        ),
    ]
    SRC_TEMPLATE = "template.html"
    SRC_IGNORE = {".ipynb_checkpoints"}
    SRC_COPY = {".png", ".svg", ".css", ".js"}

    class Error(Exception):
        def __init__(self, path, description):
            self.path = path
            super().__init__(f'Error building "{path}" - {description}')

    def __init__(self, src_root, dest_root):
        self.src_root = src_root
        self.dest_root = dest_root

    def _check_ignore(self, path):
        parts = path.split(os.path.sep)
        return any(part in self.SRC_IGNORE for part in parts) or parts[-1].startswith(
            ".#"
        )

    def _src_to_dest(self, src):
        return os.path.join(self.dest_root, os.path.relpath(src, self.src_root))

    def _render_rule(self, src):
        dest = self._src_to_dest(src).replace(".md", ".html")
        return RenderRule(
            dest, src, template=os.path.join(self.src_root, self.SRC_TEMPLATE)
        )

    def _get_rules(self, src):
        """Gets the set of rules that apply from this source path."""
        if self._check_ignore(src):
            logging.info(f"ignore {src}")
            return set([])
        ext = os.path.splitext(src)[1]
        if ext in self.SRC_COPY:
            return {CopyRule(self._src_to_dest(src), src)}
        if ext == ".md":
            return {self._render_rule(src)}
        if os.path.relpath(src, self.src_root) == self.SRC_TEMPLATE:
            return {
                self._render_rule(src)
                for src in glob.glob(f"{self.src_root}/**/*.md", recursive=True)
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
        "--debug", action="store_true", help="watch for changes and re-render"
    )
    args = parser.parse_args()

    builder = Builder(args.src, args.dest)
    builder.rebuild()
    if args.debug:
        builder.watch()
