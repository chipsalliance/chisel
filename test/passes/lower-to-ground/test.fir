; RUN: firrtl -i %s -o %s.v -X verilog -p cdt 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Top :
  module Queue :
    output out : {valid : UInt<1>, flip ready : UInt<1>}
    out.valid := UInt(1)
  module Top :
    output this : {out : {valid : UInt<1>, flip ready : UInt<1>}}
    inst queue of Queue
    this.out := queue.out
    wire w : { x : UInt<5>, flip y : UInt<5>}
    w.x := UInt(1)
    wire a : UInt<5>
    a := UInt(1)
    w.y := a
