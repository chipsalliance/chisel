; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Split Expressions
circuit Top :
   module Top :
      input p : UInt<1>
      input clk : Clock
      input a : UInt<10>
      input b : UInt<10>
      input c : UInt<10>
      
      reg out : UInt<10>,clk with :
         reset => (p,a)

      when bits(tail(sub(a,c),1),3,3) : out <= mux(eq(bits(UInt(32),4,0),UInt(13)),tail(add(a,tail(add(b,c),1)),1),tail(sub(c,b),1))

;CHECK: node GEN_0 = sub(a, c)
;CHECK: node GEN_1 = tail(GEN_0, 1)
;CHECK: node GEN_2 = bits(GEN_1, 3, 3)
;CHECK: node GEN_3 = eq(UInt("h0"), UInt("hd"))
;CHECK: node GEN_4 = add(b, c)
;CHECK: node GEN_5 = tail(GEN_4, 1)
;CHECK: node GEN_6 = add(a, GEN_5)
;CHECK: node GEN_7 = tail(GEN_6, 1)
;CHECK: node GEN_8 = sub(c, b)
;CHECK: node GEN_9 = tail(GEN_8, 1)
;CHECK: out <= mux(GEN_2, mux(GEN_3, GEN_7, GEN_9), out)   

;CHECK: Finished Split Expressions
