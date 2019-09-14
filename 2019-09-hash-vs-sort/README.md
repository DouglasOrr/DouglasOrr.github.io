# Hash vs sort -unique

Requires [Docker](https://docs.docker.com/install/) to run:

```sh
./run_benchmark.sh
# ...wait a very long time - this does runs a lot of benchmarking!
```

See `results/{timings_cpp.csv, timings_java.csv, timings_js.csv}` for results. If you want to use Jupyter to analyse the results, perhaps try:

```sh
docker run --rm -it -p 8888:8888 -v `pwd`:/home/jovyan/work -w /home/jovyan/work jupyter/scipy-notebook
```

Note that `run_benchmark.sh` builds a Docker image - if you want to free up some disk space once you're finished, run `docker rmi hashvssort`.
