; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
   module top :
      wire outs : UInt<32>[2][1]
      outs[0][0] <= UInt(1)
      outs[0][1] <= UInt(1)

      wire out : UInt<32>[2]
      out <= outs[UInt(0)]
      out[0] <= UInt(1)

; CHECK: Done!


