; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Poison :
  module Poison :
    input clk : Clock
    input reset : UInt<1>
    input index : UInt<7>
    input wmask : {x:UInt<1>, y:UInt<1>}
    input p : UInt<1>
    output out : {x : UInt<10>, y : UInt<10>}
    poison q : {x : UInt<10>, y : UInt<10>}
    mem m : 
       data-type => {x : UInt<10>, y : UInt<10>}
       depth => 128
       read-latency => 1
       write-latency => 2
       reader => r
       writer => w
       readwriter => rw
    m.r.addr <= index
    m.r.en <= UInt(1)
    m.r.clk <= clk
    m.w.addr <= index
    m.w.en <= UInt(1)
    m.w.mask <= wmask
    m.w.clk <= clk
    m.w.data <= q

    m.rw.clk <= clk
    m.rw.addr <= index
    m.rw.en <= UInt(1)
    m.rw.wmode <= UInt(1)
    m.rw.mask <= wmask
    m.rw.data <= q
    when p : 
      out <= m.r.data
    else :
      out <= q

