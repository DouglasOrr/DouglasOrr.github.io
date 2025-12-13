# [Doug's Diversions](https://douglasorr.github.io/)

Doug's personal website.

## Building the site

1. `git clone --branch source git@github.com:DouglasOrr/DouglasOrr.github.io.git`
2. Start development server with `./dev` or `./dev build --dev`
3. Add or edit an article in `src/` and update `src/index.md`
4. Build and commit with `./dev build`
5. If happy with `site/`, then `git -C site push`

Other tools:

```sh
./dev spellcheck src/path/article.md
```

**First setup**

Or use the devcontainer.

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r tools/requirements.txt
```
