"""A script to check local relative & web links from a bunch of html files."""

import argparse
import collections
import html.parser
import os
import urllib.parse
import urllib.request


Link = collections.namedtuple('Link', ('href', 'tag', 'attr', 'path', 'line', 'column'))


class LinkCheckParser(html.parser.HTMLParser):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.links = []

    def handle_starttag(self, tag, attrs):
        for attr, value in attrs:
            if attr in {'src', 'href'}:
                self.links.append(Link(value, tag, attr, self.path, *self.getpos()))

    @classmethod
    def get_links(cls, path):
        with open(path) as f:
            parser = cls(path)
            parser.feed(f.read())
            return parser.links


def check_link(link, file_name, site_files):
    # Don't check fragment-ID mapping
    href = urllib.parse.urldefrag(link.href).url
    if href.startswith('http'):
        try:
            urllib.request.urlopen(urllib.request.Request(href, method='HEAD')).close()
            return True
        except urllib.request.HTTPError as e:
            return False
    else:
        if href == '':
            local_path = file_name
        elif href == '/':
            local_path = 'index.html'
        elif href.startswith('/'):
            local_path = href[1:]
        else:
            local_path = os.path.normpath(os.path.join(os.path.dirname(file_name), href))
        return local_path in site_files


def check_links(root):
    site_files = {
        os.path.relpath(os.path.join(path, file), root)
        for path, _, files in os.walk(root)
        for file in files
    }
    broken = []
    for file_name in site_files:
        if os.path.splitext(file_name)[-1] == '.html':
            path = os.path.join(root, file_name)
            for link in LinkCheckParser.get_links(path):
                if not check_link(link, file_name, site_files):
                    broken.append(link)
    if broken:
        print('Error! broken links:')
        for link in broken:
            print(f'    <{link.tag} {link.attr}="{link.href}">,'
                  f' in {link.path}, line {link.line} column {link.column}')
    return len(broken)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', help='root path to site')
    args = parser.parse_args()
    exit(check_links(args.root))
