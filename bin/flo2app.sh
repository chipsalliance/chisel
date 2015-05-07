#!/bin/bash

flo-llvm --vcdtmp $1.flo
echo FLO-LLVM DONE
flo-llvm-release $1.flo --harness > $1-harness.cpp
echo FLO-LLVM-RELEASE DONE
g++ -o $1 $1-harness.cpp $1.o
echo GPP DONE
