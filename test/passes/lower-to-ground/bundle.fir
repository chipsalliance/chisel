; RUN: firrtl -i %s -o %s.flo -X flo -p cd | tee %s.out | FileCheck %s

circuit top :
   module m :
      input  a : { x : UInt<5>, flip y: SInt<5>}
      output b : { x : UInt<5>, flip y: SInt<5>}
   module top :
      input c : { x : UInt<5>[5], flip y : { x : UInt<5>[3], flip y : SInt<5> } }
      wire a : { x : UInt<5>, flip y : SInt<5>}
      wire b : { x : UInt<5>, flip y : SInt<5>}
      a := b
      inst i of m
      i.a := a
      b := i.b
      wire d : UInt<5>[5]

;CHECK: Lower To Ground
;CHECK: circuit top :
;CHECK:    module m :
;CHECK:       input a$x : UInt<5>
;CHECK:       output a$y : SInt<5>
;CHECK:       output b$x : UInt<5>
;CHECK:       input b$y : SInt<5>
;CHECK:       input reset : UInt<1>
;CHECK:    module top :
;CHECK:       input c$x$0 : UInt<5>
;CHECK:       input c$x$1 : UInt<5>
;CHECK:       input c$x$2 : UInt<5>
;CHECK:       input c$x$3 : UInt<5>
;CHECK:       input c$x$4 : UInt<5>
;CHECK:       output c$y$x$0 : UInt<5>
;CHECK:       output c$y$x$1 : UInt<5>
;CHECK:       output c$y$x$2 : UInt<5>
;CHECK:       input c$y$y : SInt<5>
;CHECK:       input reset : UInt<1>
;CHECK:       wire a$x : UInt<5>
;CHECK:       wire a$y : SInt<5>
;CHECK:       wire b$x : UInt<5>
;CHECK:       wire b$y : SInt<5>
;CHECK:       a$x := b$x
;CHECK:       b$y := a$y
;CHECK:       inst i of m
;CHECK:       i.reset := reset
;CHECK:       i.a$x := a$x
;CHECK:       a$y := i.a$y
;CHECK:       b$x := i.b$x
;CHECK:       i.b$y := b$y
;CHECK:       wire d$0 : UInt<5>
;CHECK:       wire d$1 : UInt<5>
;CHECK:       wire d$2 : UInt<5>
;CHECK:       wire d$3 : UInt<5>
;CHECK:       wire d$4 : UInt<5>
;CHECK: Finished Lower To Ground
