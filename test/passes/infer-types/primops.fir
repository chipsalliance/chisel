; RUN: firrtl %s abcde | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire io : {input x : UInt, output y : UInt}
      io.x := add(io.x,io.y)
;CHECK: n:io:{input x : UInt, output y : UInt}.x := add(n:io:{input x : UInt, output y : UInt}.x, n:io:{input x : UInt, output y : UInt}.y)
