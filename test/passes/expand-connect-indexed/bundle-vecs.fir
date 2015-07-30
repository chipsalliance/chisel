; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Expand Indexed Connects
circuit top :
   module top :
      wire i : UInt
      i := UInt(1)
      wire j : UInt
      j := UInt(1)

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      a[0].x := UInt(1)
      a[0].y := UInt(1)
      a[1].x := UInt(1)
      a[1].y := UInt(1)
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}y : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}y : UInt<32>


      infer accessor b = a[i]
      ; CHECK: wire b{{[_$]+}}x : UInt<32>
      ; CHECK: wire b{{[_$]+}}y : UInt<32>
      ; CHECK: b{{[_$]+}}x := a{{[_$]+}}0{{[_$]+}}x
      ; CHECK: node i!0 = i
      ; CHECK: when eq(i!0, UInt("h00000001")) : b{{[_$]+}}x := a{{[_$]+}}1{{[_$]+}}x      
      ; CHECK: node i!1 = i
      ; CHECK: when eq(i!1, UInt("h00000000")) : a{{[_$]+}}0{{[_$]+}}y := b{{[_$]+}}y      
      ; CHECK: when eq(i!1, UInt("h00000001")) : a{{[_$]+}}1{{[_$]+}}y := b{{[_$]+}}y      
      j := b.x
      b.y := UInt(1)
      
; CHECK: Finished Expand Indexed Connects

