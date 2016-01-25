; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Split Expressions
circuit Top :
   module Top :
      input p : UInt<1>
      input clk : Clock
      input a : UInt<10>
      input b : UInt<10>
      input c : UInt<10>
      
      reg out : UInt<10>,clk,p,a

      when bit(subw(a,c),3) : out <= mux(eqv(bits(UInt(32),4,0),UInt(13)),addw(a,addw(b,c)),subw(c,b))

;CHECK: node GEN_0 = subw(a, c)
;CHECK: node GEN_1 = bit(GEN_0, 3)
;CHECK: node GEN_2 = eqv(UInt("h0"), UInt("hd"))
;CHECK: node GEN_3 = addw(b, c)
;CHECK: node GEN_4 = addw(a, GEN_3)
;CHECK: node GEN_5 = subw(c, b)
;CHECK: out <= mux(GEN_1, mux(GEN_2, GEN_4, GEN_5), out)   

;CHECK: Finished Split Expressions
