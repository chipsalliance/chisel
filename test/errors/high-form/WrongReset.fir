; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s

; CHECK: Module Top has a reset that is not of type UInt<1>.
; CHECK: Module B has a reset that is not of type UInt<1>.
; CHECK: Module C has a reset that is not of type UInt<1>.

circuit Top :
  module Top :
    input reset : SInt<1>
  module B :
    input reset : UInt
  module C :
    output reset : UInt<1>


