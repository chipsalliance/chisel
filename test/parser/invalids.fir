; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
; CHECK: Done!
circuit GCD : 
   module GCD : 
      input x : UInt<128>
      input p : UInt<1>
      input q : UInt<1>
      input clk : Clock
      wire w : UInt[3]
      w is invalid
      w[0] <= UInt(0)
      w[1] <= UInt(0)
      w[2] <= UInt(0)
