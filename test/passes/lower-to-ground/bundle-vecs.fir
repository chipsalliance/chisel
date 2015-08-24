; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}y : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}y : UInt<32>

      infer accessor b = a[i]
      ; CHECK: wire b{{[_$]+}}x : UInt<32>
      ; CHECK: wire b{{[_$]+}}y : UInt<32>
      ; CHECK: b{{[_$]+}}x := (a{{[_$]+}}0{{[_$]+}}x a{{[_$]+}}1{{[_$]+}}x)[i]
      ; CHECK: (a{{[_$]+}}0{{[_$]+}}y a{{[_$]+}}1{{[_$]+}}y)[i] := b{{[_$]+}}y
      j := b
      
      infer accessor c = a[i]
      ; CHECK: wire c{{[_$]+}}x : UInt<32>
      ; CHECK: wire c{{[_$]+}}y : UInt<32>
      ; CHECK: (a{{[_$]+}}0{{[_$]+}}x a{{[_$]+}}1{{[_$]+}}x)[i] := c{{[_$]+}}x
      ; CHECK: c{{[_$]+}}y := (a{{[_$]+}}0{{[_$]+}}y a{{[_$]+}}1{{[_$]+}}y)[i]
      c := j


; CHECK: Finished Lower To Ground

