#!/bin/bash
set -ev
# Install Verilator (http://www.veripool.org/projects/verilator/wiki/Installing)
bash .install_verilator.sh
# Install firrtl
git clone --depth 10 https://github.com/freechipsproject/firrtl
cd firrtl
git checkout master
git pull
sbt +clean +publishLocal
