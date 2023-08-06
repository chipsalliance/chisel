#!/usr/bin/env bash
##===- utils/equiv-rtl.sh - Formal Equivalence via yosys------*- Script -*-===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
##===----------------------------------------------------------------------===##
#
# This script checks two input verilog files for equivalence using yosys.
#
# Usage equiv-rtl.sh File1.v File2.v TopLevelModuleName
#
##===----------------------------------------------------------------------===##
if [ "$4" != "" ]; then
    mdir=$4
else
    mdir=.
fi

echo "Comparing $1 and $2 with $3 Missing Dir $mdir"
yosys -q -p "
 read_verilog $1
 hierarchy -libdir $mdir
 rename $3 top1
 proc
 memory
 flatten top1
 read_verilog $2
 hierarchy -libdir $mdir 
 rename $3 top2
 proc
 memory
 flatten top2
 clean -purge
 opt -purge
 equiv_make top1 top2 equiv
 hierarchy -top equiv
 equiv_simple -undef
 equiv_induct -undef
 equiv_status -assert
"
if [ $? -eq 0 ]
then
  echo "PASS,INDUCT"
  exit 0
fi

#repeat with sat
echo "Trying SAT $1 and $2 with $3 Missing Dir $mdir"
yosys -q -p "
 read_verilog $1
 hierarchy -libdir $mdir
 rename $3 top1
 proc
 memory
 flatten top1
 read_verilog $2
 hierarchy -libdir $mdir 
 rename $3 top2
 proc
 memory
 flatten top2
 opt
 miter -equiv -make_assert -flatten top1 top2 equiv
 hierarchy -top equiv
 opt
 sat -prove-asserts -seq 4 -verify
"
if [ $? -eq 0 ]
then
  echo "PASS,SAT"
  exit 0
fi

echo "FAIL"
exit 1

