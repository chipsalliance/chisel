#!/usr/bin/env bash
##===- Run auditwheels with special options ------------------*- Script -*-===##
# 
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# 
##===----------------------------------------------------------------------===##
#
# Run auditwheels, but filter out all the shared objects in the 'collateral'
# directory. Those aren't run by pycde, just copied to build packages. Called by
# cibuildwheel.
#
##===----------------------------------------------------------------------===##

set -e

EXCLUDES="
  --exclude libcapnp-0.9.1.so
  --exclude libcapnp-rpc-0.9.1.so
  --exclude libkj-0.9.1.so
  --exclude libkj-async-0.9.1.so
  --exclude libkj-gzip-0.9.1.so
  --exclude libMtiPli.so
  --exclude libEsiCosimDpiServer.so"

echo auditwheel repair -w $1 $2 $EXCLUDES
auditwheel repair -w $1 $2 $EXCLUDES
