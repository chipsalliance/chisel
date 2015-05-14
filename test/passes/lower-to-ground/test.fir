; RUN: firrtl -i %s -o %s.flo -x X -p cdt | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Top :
  module Queue :
    output out : {valid : UInt<1>, flip ready : UInt<1>}
  module Top :
    output this : {out : {valid : UInt<1>, flip ready : UInt<1>}}
    inst queue of Queue
    this.out := queue.out
    wire w : { x : UInt, flip y : UInt}
    wire a : UInt
    w.y := a
