# Pages site

My personal [github site](http://douglasorr.github.io/) for random nonsense.

## Building and running the site

The cleanest way (after installing [Docker](https://www.docker.com/)):

    docker build -t blog .
    docker run -d --name blog -p 4000:4000 blog

Or if you want to run natively (e.g. on Ubuntu), e.g. for developing and auto-reloading:

    sudo apt-get install bundler
    bundle install
    bundle exec jekyll serve --drafts --watch

### Updating dependencies

We include the following dependencies:

    https://bootswatch.com/ - Yeti theme (including bootstrap CSS)
    http://getbootstrap.com/ - Bootstrap (javascript)
    https://jquery.com/download/ - JQuery

## Resources

 - [Jekyll](http://jekyllrb.com/)
 - [Jekyll templates](http://jekyllrb.com/docs/templates/)
