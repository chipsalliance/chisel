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

;CHECK: node out_1 = subw(a, c)
;CHECK: node out_2 = bit(out_1, 3)
;CHECK: node out_3 = eqv(UInt("h0"), UInt("hd"))
;CHECK: node out_4 = addw(b, c)
;CHECK: node out_5 = addw(a, out_4)
;CHECK: node out_6 = subw(c, b)
;CHECK: node out_7 = mux(out_3, out_5, out_6)

;CHECK: Finished Split Expressions
