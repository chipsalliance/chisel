; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
circuit Top : 
  module Top : 
    input clk : Clock
    input reset : UInt<1>
    input a : { w : UInt<42>, flip x : UInt<30>}[2]
    output b : { w : UInt<42>, flip x : UInt<30>}[2]
    mem m : 
       depth => 10
       data-type => UInt<32>[4]
       read-latency => 0
       write-latency => 1
       reader => r
       writer => w
       read-writer => rw
    wire x : { w : UInt<42>, x : UInt<20>}
    reg c : { w : UInt<42>, x : UInt<20>},clk,reset,x
    inst other of Other
   
    clk is invalid
    reset is invalid
    a is invalid
    b is invalid
    m is invalid
    x is invalid
    c is invalid
    other is invalid
  module Other :
    input a : { w : UInt<42>, flip x : UInt<30>}
    output b : { w : UInt<42>, flip x : UInt<30>}
    b <= a


;CHECK: Expand Connects
;CHECK: skip
;CHECK: skip
;CHECK: a[0].x is invalid
;CHECK: a[1].x is invalid
;CHECK: b[0].w is invalid
;CHECK: b[1].w is invalid
;CHECK: m.r.addr is invalid
;CHECK: m.r.en is invalid
;CHECK: m.r.clk is invalid
;CHECK: m.w.data[0] is invalid
;CHECK: m.w.data[1] is invalid
;CHECK: m.w.data[2] is invalid
;CHECK: m.w.data[3] is invalid
;CHECK: m.w.mask[0] is invalid
;CHECK: m.w.mask[1] is invalid
;CHECK: m.w.mask[2] is invalid
;CHECK: m.w.mask[3] is invalid
;CHECK: m.w.addr is invalid
;CHECK: m.w.en is invalid
;CHECK: m.w.clk is invalid
;CHECK: m.rw.rmode is invalid
;CHECK: m.rw.data[0] is invalid
;CHECK: m.rw.data[1] is invalid
;CHECK: m.rw.data[2] is invalid
;CHECK: m.rw.data[3] is invalid
;CHECK: m.rw.mask[0] is invalid
;CHECK: m.rw.mask[1] is invalid
;CHECK: m.rw.mask[2] is invalid
;CHECK: m.rw.mask[3] is invalid
;CHECK: m.rw.addr is invalid
;CHECK: m.rw.en is invalid
;CHECK: m.rw.clk is invalid
;CHECK: x.w is invalid
;CHECK: x.x is invalid
;CHECK: c.w is invalid
;CHECK: c.x is invalid
;CHECK: other.a.w is invalid
;CHECK: other.b.x is invalid   
;CHECK: Done!
