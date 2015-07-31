; RUN: firrtl -i %s -o %s.flo -X flo -p c 2>&1 | tee %s.out | FileCheck %s
;CHECK: Done!
circuit LFSR16 : 
  module LFSR16 : 
    output out : UInt<16>
    input inc : UInt<1>
    input clk : Clock
    input reset : UInt<1>
    
    reg res : UInt<16>,clk,reset
    onreset res := UInt<16>(1)
    when inc : 
      node T_16 = bit(res, 0)
      node T_17 = bit(res, 2)
      node T_18 = xor(T_16, T_17)
      node T_19 = bit(res, 3)
      node T_20 = xor(T_18, T_19)
      node T_21 = bit(res, 5)
      node T_22 = xor(T_20, T_21)
      node T_23 = bits(res, 15, 1)
      node T_24 = cat(T_22, T_23)
      res := T_24
    out := res
