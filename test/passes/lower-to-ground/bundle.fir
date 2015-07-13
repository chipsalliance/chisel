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
;CHECK:       input a_x : UInt<5>
;CHECK:       output a_y : SInt<5>
;CHECK:       output b_x : UInt<5>
;CHECK:       input b_y : SInt<5>
;CHECK:    module top :
;CHECK:       input c_x_0 : UInt<5>
;CHECK:       input c_x_1 : UInt<5>
;CHECK:       input c_x_2 : UInt<5>
;CHECK:       input c_x_3 : UInt<5>
;CHECK:       input c_x_4 : UInt<5>
;CHECK:       output c_y_x_0 : UInt<5>
;CHECK:       output c_y_x_1 : UInt<5>
;CHECK:       output c_y_x_2 : UInt<5>
;CHECK:       input c_y_y : SInt<5>
;CHECK:       wire a_x : UInt<5>
;CHECK:       wire a_y : SInt<5>
;CHECK:       wire b_x : UInt<5>
;CHECK:       wire b_y : SInt<5>
;CHECK:       a_x := b_x
;CHECK:       b_y := a_y
;CHECK:       inst i of m
;CHECK:       i.a_x := a_x
;CHECK:       a_y := i.a_y
;CHECK:       b_x := i.b_x
;CHECK:       i.b_y := b_y
;CHECK:       wire d_0 : UInt<5>
;CHECK:       wire d_1 : UInt<5>
;CHECK:       wire d_2 : UInt<5>
;CHECK:       wire d_3 : UInt<5>
;CHECK:       wire d_4 : UInt<5>
;CHECK: Finished Lower To Ground
