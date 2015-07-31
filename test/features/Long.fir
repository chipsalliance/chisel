; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done
circuit Top : 
  module Top : 
    wire a : UInt
    a := UInt("h4261441663")
