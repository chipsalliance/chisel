; RUN: firrtl -i %s -o %s.v -X verilog -p cd 2>&1 | tee %s.out | FileCheck %s
; CHECK: Expand Whens
circuit top :
   module top :
      input clk : Clock
      input reset : UInt<1>
      wire w : { x : UInt, y : UInt}
      reg r : { x : UInt, y : UInt},clk with :
         reset => (reset,w)
      wire a : UInt
      wire b : UInt
      a <= UInt(1)
      b <= UInt(2)

      w.x <= b
      w.y <= a
      r.x <= a
      r.y <= b

; CHECK: w.x <= b
; CHECK: w.y <= a
; CHECK: r.x <= a
; CHECK: r.y <= b
; CHECK: a <= UInt("h1")
; CHECK: b <= UInt("h2")
      
; CHECK: Finished Expand Whens
