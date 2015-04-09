; RUN: firrtl %s abcefghipj cg | tee %s.out | FileCheck %s 

; CHECK: Expand Whens

circuit TestLower : 
  module Inst :
     input x : UInt
     output y : UInt
  module TestLower : 
     mem m : {data : { w : UInt , x : UInt } tag : { y : UInt, z : UInt }}[8]
     wire index : UInt
     accessor r = m[index]

     inst i of Inst
     i.x := r

; CHECK: Finished Expand Whens
