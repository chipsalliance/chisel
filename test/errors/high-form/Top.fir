; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: A single module must be named Top.

circuit Top :
  module Top1 :
    wire x : UInt<1>
  module Top2 : 
    wire x : UInt<1>
