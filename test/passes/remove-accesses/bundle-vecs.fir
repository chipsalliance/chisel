; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Remove Accesses
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

      wire b : { x : UInt<32>, flip y : UInt<32> }
      b <= a[i]
      j <= b.x
      b.y <= UInt(1)
      
; CHECK: wire i : UInt<1>
; CHECK: i <= UInt("h1")
; CHECK: wire j : UInt<32>
; CHECK: j <= UInt("h1")
; CHECK: wire a : { x : UInt<32>, flip y : UInt<32>}[2]
; CHECK: a[0].x <= UInt("h1")
; CHECK: a[0].y <= UInt("h1")
; CHECK: a[1].x <= UInt("h1")
; CHECK: a[1].y <= UInt("h1")
; CHECK: wire b : { x : UInt<32>, flip y : UInt<32>}
; CHECK: wire GEN_0 : UInt<32>
; CHECK: GEN_0 <= a[0].x
; CHECK: when eqv(UInt("h1"), i) : GEN_0 <= a[1].x
; CHECK: b.x <= GEN_0
; CHECK: wire GEN_1 : UInt<32>
; CHECK: when eqv(UInt("h0"), i) : a[0].y <= GEN_1
; CHECK: when eqv(UInt("h1"), i) : a[1].y <= GEN_1
; CHECK: GEN_1 <= b.y
; CHECK: j <= b.x
; CHECK: b.y <= UInt("h1")   
; CHECK: Finished Remove Access
; CHECK: Done!

