; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: AccessIndexUInt.fir@8.4: [module Top]  Access index must be a UInt type.
circuit Top : 
  module Top : 
    input b : UInt<32>[62]
    input a : SInt<8>
    output c : UInt<32>
    c <= b[a]
