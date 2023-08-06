#!/usr/bin/env bash
##===- utils/get-capnp.sh - Install CapnProto ----------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# This script downloads, compiles, and installs CapnProto into $/ext.
# Cap'nProto is use by ESI (Elastic Silicon Interfaces) cosimulation as a
# message format and RPC client/server.
#
# It will also optionally install pycapnp, which is used for testing.
#
##===----------------------------------------------------------------------===##

echo "Do you wish to install pycapnp? Cosim integration tests require pycapnp."
read -p "Yes to confirm: " yn
case $yn in
    [Yy]* ) pip3 install pycapnp;;
    * ) echo "Skipping.";;
esac

mkdir -p "$(dirname "$BASH_SOURCE[0]")/../ext"
EXT_DIR=$(cd "$(dirname "$BASH_SOURCE[0]")/../ext" && pwd)
CAPNP_VER=0.9.1
echo "Installing capnproto..."

echo $EXT_DIR
cd $EXT_DIR

wget https://capnproto.org/capnproto-c++-$CAPNP_VER.tar.gz
tar -zxf capnproto-c++-$CAPNP_VER.tar.gz
cd capnproto-c++-$CAPNP_VER
./configure --prefix=$EXT_DIR
make -j$(nproc)
make install
cd ../
rm -r capnproto-c++-$CAPNP_VER capnproto-c++-$CAPNP_VER.tar.gz

echo "Done."
