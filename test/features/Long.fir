; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s
;CHECK: Done
circuit Top : 
  module Top : 
    wire a : UInt
    a := UInt(4261441663)
