import glob
import markdown
import os
import re
import logging


class ResolveLinks(markdown.treeprocessors.Treeprocessor):
    def run(self, root):
        for link in root.iter('a'):
            href = link.attrib['href']
            if href.endswith('.md'):
                link.attrib['href'] = re.sub('.md$', '.html', href)


class DougExtension(markdown.extensions.Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(ResolveLinks(md), 'resolve_links', 20)


if __name__ == '__main__':
    src_root = 'src'
    site_root = 'site'
    logging.basicConfig(level=logging.INFO)
    os.makedirs(site_root, exist_ok=True)
    for src in glob.glob(f'{src_root}/*.md'):
        src_name = os.path.splitext(os.path.relpath(src, src_root))[0]
        dest = os.path.join(site_root, f'{src_name}.html')
        logging.info(f'{src} => {dest}')
        markdown.markdownFromFile(
            input=src,
            output=dest,
            extensions=[DougExtension()]
        )

# print(markdown.markdown("""
# # Title

# Some [custom link](foo.md).

# [Another link](bar/baz.md).
# """, extensions=[DougExtension()]))
