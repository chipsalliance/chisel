#!/bin/bash
set -ex

git config --global user.email "schuyler.eldridge@gmail.com"
git config --global user.name "seldridge"
git config --global push.default simple

(
  cd $HOME
  git clone --branch gh-pages git@github.com:freechipsproject/www.chisel-lang.org site
  cd site
  rm -rf *
  cp -r $TRAVIS_BUILD_DIR/docs/target/site/* .
  git add .
  git commit -m "Published from $TRAVIS_COMMIT"
  git push origin gh-pages
)
