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

    # Clean
    for old in glob.glob(f'{site_root}/*.html'):
        logging.info(f'remove {old}')
        os.remove(old)

    # Build
    md = markdown.Markdown(extensions=[DougExtension()])
    for src in glob.glob(f'{src_root}/*.md'):
        src_name = os.path.splitext(os.path.relpath(src, src_root))[0]
        dest = os.path.join(site_root, f'{src_name}.html')
        logging.info(f'{src} => {dest}')
        md.convertFile(src, dest)
