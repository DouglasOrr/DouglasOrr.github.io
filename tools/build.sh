TOOLS_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
echo $TOOLS_DIR
set -e

(git -C site fetch && git -C site reset --hard origin/master) || git clone git@github.com:DouglasOrr/DouglasOrr.github.io.git site/

docker build --rm -t pages-douglasorr "$TOOLS_DIR"

docker run --rm -it -v "$(dirname $TOOLS_DIR)":/work -w /work --user "$(id -u):$(id -g)" \
       pages-douglasorr \
       python3 tools/render.py

git -C site add . && git -C site commit -m 'Publish'
