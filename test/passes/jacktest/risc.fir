; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s
; CHECK: Done!
circuit Risc : 
  module Risc : 
    output out : UInt<32>
    output valid : UInt<1>
    input boot : UInt<1>
    input isWr : UInt<1>
    input wrAddr : UInt<8>
    input wrData : UInt<32>
    input clk : Clock
    input reset : UInt<1>
    
    cmem file : UInt<32>[256],clk
    cmem code : UInt<32>[256],clk
    reg pc : UInt<8>,clk,reset
    onreset pc <= UInt<8>(0)
    infer accessor inst = code[pc]
    node op = bits(inst, 31, 24)
    node rci = bits(inst, 23, 16)
    node rai = bits(inst, 15, 8)
    node rbi = bits(inst, 7, 0)
    node T_51 = eq(rai, UInt<1>(0))
    infer accessor T_52 = file[rai]
    node ra = mux(T_51, UInt<1>(0), T_52)
    node T_53 = eq(rbi, UInt<1>(0))
    infer accessor T_54 = file[rbi]
    node rb = mux(T_53, UInt<1>(0), T_54)
    wire rc : UInt<32>
    valid <= UInt<1>(0)
    out <= UInt<1>(0)
    rc <= UInt<1>(0)
    when isWr : 
      infer accessor T_55 = code[wrAddr]
      T_55 <= wrData
    else : when boot : pc <= UInt<1>(0)
      else : 
        node T_56 = eq(UInt<1>(0), op)
        when T_56 : 
          node T_57 = addw(ra, rb)
          rc <= T_57
        node T_58 = eq(UInt<1>(1), op)
        when T_58 : 
          node T_59 = shl(rai, 8)
          node T_60 = or(T_59, rbi)
          rc <= T_60
        out <= rc
        node T_61 = eq(rci, UInt<8>(255))
        when T_61 : valid <= UInt<1>(1)
        else : 
          infer accessor T_62 = file[rci]
          T_62 <= rc
        node T_63 = addw(pc, UInt<1>(1))
        pc <= T_63
