#!/bin/bash
set -ev
git clone --depth 10 https://github.com/freechipsproject/firrtl
cd firrtl
git checkout master
git pull
FIRRTL_SHA1=`git rev-parse HEAD`
BUILT_FIRRTL_SHA1=""
if [ -s $HOME/.ivy2/firrtl.sha1 ]; then
  BUILT_FIRRTL_SHA1=`cat $HOME/.ivy2/firrtl.sha1`
fi
if [ "$FIRRTL_SHA1" != "$BUILT_FIRRTL_SHA"1 ]; then
  sbt +clean +publishLocal
  echo $FIRRTL_SHA1 > $HOME/.ivy2/firrtl.sha1
fi

