; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      output out : UInt<64>
      input index : UInt<1>
      wire T_292 : UInt<64>[2]
      T_292[0] := UInt(1)
      T_292[1] := UInt(1)
      infer accessor T_297 = T_292[index]
      out := T_297
;CHECK: Done!
