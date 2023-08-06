#!/usr/bin/env bash
##===- utils/get-verilator.sh - Install Verilator ------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Downloads, compiles, and installs Verilator into $/ext
# Verilator can be used to check SystemVerilog code.
#
##===----------------------------------------------------------------------===##

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
VERILATOR_VER=4.110

echo $EXT_DIR
cd $EXT_DIR

wget https://github.com/verilator/verilator/archive/v$VERILATOR_VER.tar.gz
tar -zxf v$VERILATOR_VER.tar.gz
cd verilator-$VERILATOR_VER
autoconf
./configure --prefix=$EXT_DIR
make -j$(nproc)
make install
cd ..
rm -r verilator-$VERILATOR_VER v$VERILATOR_VER.tar.gz
