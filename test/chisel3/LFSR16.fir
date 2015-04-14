circuit LFSR16 : 
  module LFSR16 : 
    output out : UInt(16)
    input inc : UInt(1)
    
    node T_16 : UInt(16) = UInt(1, 16)
    reg res : UInt(16)
    res.init := T_16
    when inc : 
      node T_17 : UInt(1) = bit(res, 0)
      node T_18 : UInt(1) = bit(res, 2)
      node T_19 : UInt(1) = bit-xor(T_17, T_18)
      node T_20 : UInt(1) = bit(res, 3)
      node T_21 : UInt(1) = bit-xor(T_19, T_20)
      node T_22 : UInt(1) = bit(res, 5)
      node T_23 : UInt(1) = bit-xor(T_21, T_22)
      node T_24 : UInt = bits(res, 15, 1)
      node T_25 : UInt(1) = concat(T_23, T_24)
      res := T_25
    out := res