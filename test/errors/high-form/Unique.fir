; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
; XFAIL: *

; CHECK: Reference x does not have a unique name.
; CHECK: Reference p does not have a unique name.
; CHECK-NOT: Reference q does not have a unique name.

circuit Top :
  module Top :
    wire x : UInt<1>
    wire x : UInt<2>
    wire p : UInt<3>
    wire q : UInt<3>
    when p :
       wire p : UInt<4>
  module Other :
     wire q : UInt<3>
