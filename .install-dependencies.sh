#!/bin/bash
set -ev
# Install firrtl
git clone --depth 10 https://github.com/freechipsproject/firrtl
cd firrtl
git checkout master
git pull
sbt clean publishLocal
