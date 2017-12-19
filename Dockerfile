FROM ruby:2.3-onbuild
CMD bundle exec jekyll serve -H 0.0.0.0
