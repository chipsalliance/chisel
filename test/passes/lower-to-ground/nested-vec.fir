; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
circuit top :
   module top :
      input clk : Clock
      wire i : UInt
      wire j : { x : UInt<32>, flip y : UInt<32> }
      wire k : { x : UInt<32>, y : UInt<32> }

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

      cmem m : { x : UInt<32>, y : UInt<32> }[2],clk
      ; CHECK: cmem m{{[_$]+}}x : UInt<32>[2]
      ; CHECK: cmem m{{[_$]+}}y : UInt<32>[2]

      infer accessor c = m[i] ; MALE
      ; CHECK: accessor c{{[_$]+}}x = m{{[_$]+}}x[i]
      ; CHECK: accessor c{{[_$]+}}y = m{{[_$]+}}y[i]

      c := k
      ; CHECK: c{{[_$]+}}x := k{{[_$]+}}x
      ; CHECK: c{{[_$]+}}y := k{{[_$]+}}y

; CHECK: Finished Lower To Ground

