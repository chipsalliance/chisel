
; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: A single module must be named Top.

circuit Top :
  module Top1 :
    wire x : UInt<1>
  module Top2 : 
    wire x : UInt<1>
