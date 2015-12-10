; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Inline Indexers
circuit top :
   module top :
      wire i : UInt
      i <= UInt(1)
      wire j : UInt
      j <= UInt(1)

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      a[0].x <= UInt(1)
      a[0].y <= UInt(1)
      a[1].x <= UInt(1)
      a[1].y <= UInt(1)
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}y : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}y : UInt<32>


      infer accessor b = a[i]
      ; CHECK: wire b{{[_$]+}}x_2 : UInt<32>
      ; CHECK: node i_1 = i
      ; CHECK: b{{[_$]+}}x_2 <= a{{[_$]+}}0{{[_$]+}}x
      ; CHECK: when eqv(i_1, UInt("h1")) : b{{[_$]+}}x_2 <= a{{[_$]+}}1{{[_$]+}}x      
      ; CHECK: wire b{{[_$]+}}y_2 : UInt<32>
      ; CHECK: node i_2 = i
      ; CHECK: when eqv(i_2, UInt("h0")) : a{{[_$]+}}0{{[_$]+}}y <= b{{[_$]+}}y_2
      ; CHECK: when eqv(i_2, UInt("h1")) : a{{[_$]+}}1{{[_$]+}}y <= b{{[_$]+}}y_2
      j <= b.x
      b.y <= UInt(1)
      
; CHECK: Finished Inline Indexers
; CHECK: Done!

