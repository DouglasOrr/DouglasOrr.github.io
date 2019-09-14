trap "kill 0" EXIT
set -e

TOOLS_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)
SITE="site_debug"
PORT="8000"

mkdir -p "${SITE}"
(cd "${SITE}" && python3 -m http.server "${PORT}" &> server.log &)
echo
echo "Started dev server at: http://localhost:${PORT}"
echo

docker build --rm -t pages-douglasorr "${TOOLS_DIR}"

docker run --rm -it -v "$(dirname ${TOOLS_DIR})":/work -w /work --user "$(id -u):$(id -g)" \
       pages-douglasorr \
       python3 tools/render.py src/ ${SITE} --debug
