; RUN: firrtl %s abcde | tee %s.out | FileCheck %s

;CHECK: Infer Types
circuit top :
   module top :
      wire io : {input x : UInt, output y : UInt}
      wire a : UInt 
      a := io.x

