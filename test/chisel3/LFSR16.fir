; RUN: firrtl -i %s -o %s.flo -x X -p c | tee %s.out | FileCheck %s
; CHECK: Done!

circuit LFSR16 : 
  module LFSR16 : 
    output out : UInt<16>
    input inc : UInt<1>
    
    node T_16 = UInt<16>(1)
    reg res : UInt<16>
    on-reset res := T_16
    when inc : 
      node T_17 = bit(res, 0)
      node T_18 = bit(res, 2)
      node T_19 = bit-xor(T_17, T_18)
      node T_20 = bit(res, 3)
      node T_21 = bit-xor(T_19, T_20)
      node T_22 = bit(res, 5)
      node T_23 = bit-xor(T_21, T_22)
      node T_24 = bits(res, 15, 1)
      node T_25 = cat(T_23, T_24)
      res := T_25
    out := res

