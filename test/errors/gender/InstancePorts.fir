; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK-NOT: Expression queue.in is used as a sink but can only be used as a source.
; CHECK: Done!

circuit BTB :
  module Queue :
    input in : UInt<1>
    output out : UInt<1>
    out := in
  module BTB : 
    input time : UInt<1>
    output out : UInt<1>
    
    inst queue of Queue
    queue.in := time
    out := queue.in

