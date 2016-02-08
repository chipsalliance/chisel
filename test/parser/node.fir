; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
; CHECK: Done!
circuit GCD : 
   module GCD : 
      input x : UInt<128>
      input p : UInt<1>
      input q : UInt<1>
      input clk : Clock
      reg addr : UInt, clk with :         
         reset => (UInt<1>("h0"), addr)
      when p :
         node T_1234 = bits(x, 63, 24)
         addr <= T_1234
      when q :
         node T_1380 = tail(x, 1)
         addr <= T_1380
