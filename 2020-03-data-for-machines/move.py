import os


def move(src, dest):
    """Rename a file or directory src -> dest."""
    try:
        os.rename(src, dest)
    except OSError as e:
        print(f'Error! could not move {src} to {dest}: {e}')
        exit(1)


if __name__ == '__main__':
    import sys
    move(sys.argv[1], sys.argv[2])
