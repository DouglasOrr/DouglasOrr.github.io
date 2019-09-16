# [Doug's Diversions](https://douglasorr.github.io/)

Doug's personal website.

## Building the site

 1. `git clone --branch source git@github.com:DouglasOrr/DouglasOrr.github.io.git`
 1. Add or edit an article in `src/`
 1. If needed, update `src/index.md`
 1. Run `./tools/build.sh`
 1. If happy with `site/`: `git -C site push`

## Developing the site

```sh
./tools/build_debug.sh
```