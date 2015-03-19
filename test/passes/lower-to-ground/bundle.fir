; RUN: firrtl %s abcdefgh c | tee %s.out | FileCheck %s

circuit top :
   module m :
      input  a : { x : UInt, flip y: SInt}
      output b : { x : UInt, flip y: SInt}
   module subtracter :
      input c : { x : UInt[5], flip y : { x : UInt[3], flip y : SInt } }
      wire a : { x : UInt, flip y : SInt}
      wire b : { x : UInt, flip y : SInt}
      a := b
      inst i of m
      i.a := a
      b := i.b
      wire d : UInt[5]

;CHECK: Lower To Ground
;CHECK: circuit top :
;CHECK:    module m :
;CHECK:       input a#x : UInt
;CHECK:       output a#y : SInt
;CHECK:       output b#x : UInt
;CHECK:       input b#y : SInt
;CHECK:       input reset : UInt(1)
;CHECK:    module subtracter :
;CHECK:       input c#x#0 : UInt
;CHECK:       input c#x#1 : UInt
;CHECK:       input c#x#2 : UInt
;CHECK:       input c#x#3 : UInt
;CHECK:       input c#x#4 : UInt
;CHECK:       output c#y#x#0 : UInt
;CHECK:       output c#y#x#1 : UInt
;CHECK:       output c#y#x#2 : UInt
;CHECK:       input c#y#y : SInt
;CHECK:       input reset : UInt(1)
;CHECK:       wire a#x : UInt
;CHECK:       wire a#y : SInt
;CHECK:       wire b#x : UInt
;CHECK:       wire b#y : SInt
;CHECK:       a#x := b#x
;CHECK:       b#y := a#y
;CHECK:       inst i of m
;CHECK:       i.reset := reset
;CHECK:       i.a#x := a#x
;CHECK:       a#y := i.a#y
;CHECK:       b#x := i.b#x
;CHECK:       i.b#y := b#y
;CHECK:       wire d#0 : UInt
;CHECK:       wire d#1 : UInt
;CHECK:       wire d#2 : UInt
;CHECK:       wire d#3 : UInt
;CHECK:       wire d#4 : UInt
;CHECK: Finished Lower To Ground
