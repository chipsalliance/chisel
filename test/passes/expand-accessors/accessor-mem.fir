; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

circuit top :
   module top :
      input clk : Clock
      wire i : UInt<4>
      wire j : UInt<32>
      wire z : UInt<32>
      i <= UInt(1)
      mem m : 
         data-type => UInt<32>[2]
         depth => 2
         reader => a
         writer => x
         read-latency => 0
         write-latency => 1
      m.a.addr <= i
      m.a.clk <= clk
      m.a.en <= UInt(1)
      m.x.addr <= i
      m.x.clk <= clk
      m.x.en <= UInt(1)
      m.x.mask[0] <= UInt(1)
      m.x.mask[1] <= UInt(1)
      wire b : UInt<32>[2]
      b <= m.a.data
      node c = b[i]
      z <= j
      j <= c
      wire y : UInt<32>[2]
      y[0] <= UInt(1)
      y[1] <= UInt(1)
      m.x.data <= y
      y[i] <= z

; CHECK: Done!
