#!/bin/bash
set -e

git config --global user.email "schuyler.eldridge@gmail.com"
git config --global user.name "seldridge"
git config --global push.default simple

(
  cd $HOME
  git clone --branch gh-pages git@github.com:freechipsproject/www.chisel-lang.org site
  cd site
  rm -rf *
  cp $TRAVIS_BUILD_DIR/docs/target/site/* .
  git commit -m "Published from $TRAVIS_COMMIT"
  git push origin gh-pages
)
