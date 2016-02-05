; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
circuit GCD : 
  module GCD : 
    input e : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    output z : UInt
    output v : UInt<1>
    input a : UInt<16>
    input b : UInt<16>
    
    reg x : UInt<16>,clk
    reg y : UInt<16>,clk
    node T_17 = gt(x, y)
    when T_17 : 
      node T_18 = tail(sub(x, y), 1)
      x <= T_18
    else : 
      node T_19 = tail(sub(y, x), 1)
      y <= T_19
    when e : 
      x <= a
      y <= b
    z <= x
    node T_20 = eq(y, UInt<1>(0))
    v <= T_20

; CHECK: circuit GCD : 
; CHECK:   module GCD : 
; CHECK:     input e : UInt<1>
; CHECK:     input clk : Clock
; CHECK:     input reset : UInt<1>
; CHECK:     output z : UInt<16>
; CHECK:     output v : UInt<1>
; CHECK:     input a : UInt<16>
; CHECK:     input b : UInt<16>
; CHECK:     reg x : UInt<16>, clk with : 
; CHECK:       reset => (UInt<1>("h0"), x)
; CHECK:     reg y : UInt<16>, clk with : 
; CHECK:       reset => (UInt<1>("h0"), y)
; CHECK:     node T_17 = gt(x, y)
; CHECK:     when T_17 : 
; CHECK:       node T_18 = tail(sub(x, y), 1)
; CHECK:       x <= T_18
; CHECK:     else :
; CHECK:       node T_19 = tail(sub(y, x), 1)
; CHECK:       y <= T_19
; CHECK:     when e : 
; CHECK:       x <= a
; CHECK:       y <= b
; CHECK:     z <= x
; CHECK:     node T_20 = eq(y, UInt<1>("h0"))
; CHECK:     v <= T_20

