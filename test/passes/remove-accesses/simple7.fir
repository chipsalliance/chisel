; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
   module top :
      output out : UInt<64>
      input index : UInt<1>
      wire T_292 : UInt<64>[2]
      T_292[0] <= UInt(1)
      T_292[1] <= UInt(1)
      out <= T_292[index]
;CHECK: Done!
