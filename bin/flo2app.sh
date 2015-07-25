#!/bin/bash

flo-llvm $1.flo #  --vcdtmp
if [ $? ] ; then
  echo FLO-LLVM DONE
  flo-llvm-release $1.flo --harness > $1-harness.cpp
fi
if [ $? ] ; then
  echo FLO-LLVM-RELEASE DONE
  clang++ -o $1 $1-harness.cpp $1.o
  echo GPP DONE
fi
