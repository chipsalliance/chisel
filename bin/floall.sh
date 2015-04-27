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
    echo -n $file "-> "
    (cd generated; ../bin/fir2flo.sh $file > $file.out; grep "res = " $file.out)
done
