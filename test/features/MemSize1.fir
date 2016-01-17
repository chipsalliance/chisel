; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit Top : 
  module Top : 
    input clk : Clock
    output out : UInt<32>
    input i : UInt<1>
    cmem m : UInt<32>[1]
    read mport r = m[i],clk
    out <= r

