; RUN: firrtl -i %s -o %s.flo -X flo -p cd 2>&1 | tee %s.out | FileCheck %s

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
;CHECK:       input a{{[_$]+}}x : UInt<5>
;CHECK:       output a{{[_$]+}}y : SInt<5>
;CHECK:       output b{{[_$]+}}x : UInt<5>
;CHECK:       input b{{[_$]+}}y : SInt<5>
;CHECK:    module top :
;CHECK:       input c{{[_$]+}}x{{[_$]+}}0 : UInt<5>
;CHECK:       input c{{[_$]+}}x{{[_$]+}}1 : UInt<5>
;CHECK:       input c{{[_$]+}}x{{[_$]+}}2 : UInt<5>
;CHECK:       input c{{[_$]+}}x{{[_$]+}}3 : UInt<5>
;CHECK:       input c{{[_$]+}}x{{[_$]+}}4 : UInt<5>
;CHECK:       output c{{[_$]+}}y{{[_$]+}}x{{[_$]+}}0 : UInt<5>
;CHECK:       output c{{[_$]+}}y{{[_$]+}}x{{[_$]+}}1 : UInt<5>
;CHECK:       output c{{[_$]+}}y{{[_$]+}}x{{[_$]+}}2 : UInt<5>
;CHECK:       input c{{[_$]+}}y{{[_$]+}}y : SInt<5>
;CHECK:       wire a{{[_$]+}}x : UInt<5>
;CHECK:       wire a{{[_$]+}}y : SInt<5>
;CHECK:       wire b{{[_$]+}}x : UInt<5>
;CHECK:       wire b{{[_$]+}}y : SInt<5>
;CHECK:       a{{[_$]+}}x := b{{[_$]+}}x
;CHECK:       b{{[_$]+}}y := a{{[_$]+}}y
;CHECK:       inst i of m
;CHECK:       i.a{{[_$]+}}x := a{{[_$]+}}x
;CHECK:       a{{[_$]+}}y := i.a{{[_$]+}}y
;CHECK:       b{{[_$]+}}x := i.b{{[_$]+}}x
;CHECK:       i.b{{[_$]+}}y := b{{[_$]+}}y
;CHECK:       wire d{{[_$]+}}0 : UInt<5>
;CHECK:       wire d{{[_$]+}}1 : UInt<5>
;CHECK:       wire d{{[_$]+}}2 : UInt<5>
;CHECK:       wire d{{[_$]+}}3 : UInt<5>
;CHECK:       wire d{{[_$]+}}4 : UInt<5>
;CHECK: Finished Lower To Ground
