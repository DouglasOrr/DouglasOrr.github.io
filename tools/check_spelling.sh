set -e

TARGET="${1}"
TOOLS_DIR=$(cd $(dirname "${BASH_SOURCE[0]}") && pwd)

docker build --rm -t pages-douglasorr "${TOOLS_DIR}"

docker run --rm -it -v "$(dirname ${TOOLS_DIR})":/work -w /work --user "$(id -u):$(id -g)" \
       pages-douglasorr \
       aspell --home-dir=tools/ check "${TARGET}"
