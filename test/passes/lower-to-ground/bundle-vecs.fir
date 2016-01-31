; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower Types
circuit top :
   module top :
      input i : UInt<1>
      wire j : { x : UInt<32>, flip y : UInt<32> }

      wire a : { x : UInt<32>, flip y : UInt<32> }[2]
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}0{{[_$]+}}y : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}x : UInt<32>
      ; CHECK: wire a{{[_$]+}}1{{[_$]+}}y : UInt<32>
      a[0].x <= UInt(0)
      a[0].y <= UInt(0)
      a[1].x <= UInt(0)
      a[1].y <= UInt(0)

      j <= a[i]
      a[i] <= j

; CHECK: wire GEN_0 : UInt<32>
; CHECK: wire GEN_1 : UInt<32>
; CHECK: wire GEN_2 : UInt<32>
; CHECK: wire GEN_3 : UInt<32>
; CHECK: j_x <= GEN_0
; CHECK: j_y <= GEN_3
; CHECK: a_0_x <= mux(eq(UInt<1>("h0"), i), GEN_2, UInt<1>("h0"))
; CHECK: a_0_y <= mux(eq(UInt<1>("h0"), i), GEN_1, UInt<1>("h0"))
; CHECK: a_1_x <= mux(eq(UInt<1>("h1"), i), GEN_2, UInt<1>("h0"))
; CHECK: a_1_y <= mux(eq(UInt<1>("h1"), i), GEN_1, UInt<1>("h0"))
; CHECK: GEN_0 <= mux(eq(UInt<1>("h1"), i), a_1_x, a_0_x)
; CHECK: GEN_1 <= j_y
; CHECK: GEN_2 <= j_x
; CHECK: GEN_3 <= mux(eq(UInt<1>("h1"), i), a_1_y, a_0_y)   

; CHECK: Finished Lower Types

