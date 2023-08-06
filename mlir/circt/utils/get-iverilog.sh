#!/usr/bin/env bash
##===- utils/get-iverilog.sh - Install Icarus Verilog ---------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Downloads, compiles, and installs Icarus Verilog into $/ext
# Icarus Verilog can be used to check SystemVerilog code.
#
##===----------------------------------------------------------------------===##

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
IVERILOG_VER=11_0

echo $EXT_DIR
cd $EXT_DIR

wget https://github.com/steveicarus/iverilog/archive/refs/tags/v${IVERILOG_VER}.tar.gz
tar -zxf v$IVERILOG_VER.tar.gz
cd iverilog-$IVERILOG_VER
autoconf
./configure --prefix=$EXT_DIR
make -j$(nproc)
make install
cd ..
rm -r iverilog-$IVERILOG_VER v$IVERILOG_VER.tar.gz
