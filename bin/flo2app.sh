#!/bin/bash

echo FLO-LLVM DONE
/users/jrb/bar/chisel3/bin/flo-llvm $1.flo #  --vcdtmp
echo FLO-LLVM DONE
flo-llvm-release $1.flo --harness > $1-harness.cpp
echo FLO-LLVM-RELEASE DONE
$CXX -o $1 $1-harness.cpp $1.o
echo GPP DONE
