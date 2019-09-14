import argparse
import inotify.adapters
import logging
import markdown
import os
import re
import shutil


class ResolveLinks(markdown.treeprocessors.Treeprocessor):
    def run(self, root):
        for link in root.iter('a'):
            href = link.attrib['href']
            if href.endswith('.md'):
                link.attrib['href'] = re.sub('.md$', '.html', href)


class DougExtension(markdown.extensions.Extension):
    def extendMarkdown(self, md):
        md.treeprocessors.register(ResolveLinks(md), 'resolve_links', 20)


class Builder:
    DEST_NOCLEAN = {'.git', 'README.md', 'server.log'}
    SRC_IGNORE = {'.ipynb_checkpoints'}
    SRC_COPY = {'img'}
    MARKDOWN = markdown.Markdown(extensions=[DougExtension()])

    class Error(Exception):
        def __init__(self, path, description):
            self.path = path
            super().__init__(f'Error building "{path}" - {description}')

    def __init__(self, src_root, dest_root):
        self.src_root = src_root
        self.dest_root = dest_root

    def _check_ignore(self, path):
        parts = path.split(os.path.sep)
        return any(part in self.SRC_IGNORE for part in parts) or parts[-1].startswith('.#')

    def _update_file(self, path):
        src = os.path.join(self.src_root, path)
        if self._check_ignore(path):
            logging.info(f'ignore {src}')
            return
        dest = os.path.join(self.dest_root, path)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        parts = path.split(os.path.sep)
        if any(part in self.SRC_COPY for part in parts):
            logging.info(f'copy {src} => {dest}')
            shutil.copyfile(src, dest)
        elif parts[-1].endswith('.md'):
            dest_html = dest.replace('.md', '.html')
            logging.info(f'render {src} => {dest_html}')
            self.MARKDOWN.convertFile(src, dest_html)
        else:
            raise self.Error(src, 'missing build rule')

    def _delete_file(self, path):
        if self._check_ignore(path):
            return
        dest = os.path.join(self.dest_root, path)
        if os.path.isfile(dest):
            logging.info(f'delete {dest}')
            os.remove(dest)
        else:
            raise self.Error(dest, 'cannot delete - not a file')

    def clean(self):
        os.makedirs(self.dest_root, exist_ok=True)
        for name in os.listdir(self.dest_root):
            if name not in self.DEST_NOCLEAN:
                path = os.path.join(self.dest_root, name)
                logging.info(f'clean {path}')
                if os.path.isfile(path):
                    os.remove(path)
                else:
                    shutil.rmtree(path)

    def rebuild(self):
        self.clean()
        for parent, _, files in os.walk(self.src_root):
            for file in files:
                self._update_file(os.path.relpath(os.path.join(parent, file), self.src_root))

    def watch(self):
        events = inotify.adapters.InotifyTree(self.src_root).event_gen(yield_nones=False)
        for _, types, parent, name in events:
            try:
                if 'IN_ISDIR' not in types:
                    path = os.path.relpath(os.path.join(parent, name), self.src_root)
                    if 'IN_CLOSE_WRITE' in types:
                        self._update_file(path)
                    if 'IN_DELETE' in types:
                        self._delete_file(path)
            except self.Error as e:
                logging.error(e)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description='Site render script')
    parser.add_argument('src')
    parser.add_argument('dest')
    parser.add_argument('--debug', action='store_true', help='watch for changes and re-render')
    args = parser.parse_args()

    builder = Builder(args.src, args.dest)
    builder.rebuild()
    if args.debug:
        builder.watch()
