; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Expand Connects
circuit Top : 
  module Top : 
    input a: {w:UInt<42>,x:UInt<20>}
    input b: {w:UInt<42>,x:UInt<20>}
    input c: {w:UInt<42>,x:UInt<20>}
    input p : UInt<1>
    output d: {w:UInt<42>,x:UInt<20>}
    d <= mux(p,mux(p,a,b),c)
;CHECK: d.w <= mux(p, mux(p, a.w, b.w), c.w)
;CHECK: d.x <= mux(p, mux(p, a.x, b.x), c.x)   

;CHECK: Finished Expand Connects
;CHECK: Done!

