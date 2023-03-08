set -e

TOOLS_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
SITE="site"

(git -C "${SITE}" fetch && git -C "${SITE}" reset --hard origin/main) || git clone git@github.com:DouglasOrr/DouglasOrr.github.io.git "${SITE}"

docker build --rm -t pages-douglasorr "${TOOLS_DIR}"

docker run --rm -it -v "$(dirname ${TOOLS_DIR})":/work -w /work --user "$(id -u):$(id -g)" \
       pages-douglasorr \
       python3 tools/render.py src/ "${SITE}"

touch "${SITE}/.nojekyll" && git -C "${SITE}" add . && git -C "${SITE}" commit -m 'Publish'
