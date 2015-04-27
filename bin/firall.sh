#!/bin/bash
FILES="BitsOps \
  BundleWire \
  ComplexAssign \
  Counter \
  DirChange \
  EnableShiftRegister \
  GCD \
  LFSR16 \
  MemorySearch \
  ModuleVec \
  Mul \
  Outer \
  RegisterVecShift \
  Risc \
  Rom \
  SIntOps \
  Stack \
  Tbl \
  UIntOps \
  VecApp \
  VecShiftRegister \
  VendingMachine"
          
for file in $FILES; do
    sbt "run $file"
done
