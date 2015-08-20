; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Constant Propagation
;CHECK: x := UInt("h1f")
;CHECK: y := SInt("h20")
;CHECK: Finished Constant Propagation

circuit top :
   module top :
      output x : UInt
      output y : SInt
      x := shr(UInt(127),2)
      y := shr(SInt(-128),2)

