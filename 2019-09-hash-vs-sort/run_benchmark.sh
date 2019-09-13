set -e

docker build --rm -t hashvssort .

docker run --rm -it -v `pwd`:/work -w /work --user $(id -u):$(id -g) \
       hashvssort \
       sh benchmark.sh
