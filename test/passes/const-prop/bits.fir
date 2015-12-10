; RUN: firrtl -i %s -o %s.v -X verilog -p cT 2>&1 | tee %s.out | FileCheck %s

;CHECK: Constant Propagation
;CHECK: node x = UInt("h7")
;CHECK: Finished Constant Propagation

circuit top :
   module top :
      output out : UInt
      node x = bits(UInt(127),2,0)
      out <= x
