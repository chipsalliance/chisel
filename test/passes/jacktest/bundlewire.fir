; RUN: firrtl -i %s -o %s.flo -x X -p ct | tee %s.out | FileCheck %s

; CHECK: Expand Whens

circuit TestLower : 
  module Inst :
     input data : { w : UInt , x : UInt } 
     input tag : { y : UInt, z : UInt }
  module TestLower : 
     mem m : {data : { w : UInt , x : UInt } tag : { y : UInt, z : UInt }}[8]
     wire index : UInt
     accessor r = m[index]

     inst i of Inst
     i.data := r.data

; CHECK: Finished Expand Whens
