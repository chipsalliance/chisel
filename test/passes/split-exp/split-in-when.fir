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

;CHECK: node GEN_0 = subw(a, c)
;CHECK: node GEN_1 = bits(GEN_0, 3, 3)
;CHECK: node GEN_2 = eq(UInt("h0"), UInt("hd"))
;CHECK: node GEN_3 = addw(b, c)
;CHECK: node GEN_4 = addw(a, GEN_3)
;CHECK: node GEN_5 = subw(c, b)
;CHECK: out <= mux(GEN_1, mux(GEN_2, GEN_4, GEN_5), out)   

;CHECK: Finished Split Expressions
