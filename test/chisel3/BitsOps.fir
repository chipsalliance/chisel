circuit BitsOps : 
  module BitsOps : 
    input b : UInt(16)
    input a : UInt(16)
    output notout : UInt(16)
    output andout : UInt(16)
    output orout : UInt(16)
    output xorout : UInt(16)
    
    node T_13 : UInt = bit-not(a)
    notout := T_13
    node T_14 : UInt = bit-and(a, b)
    andout := T_14
    node T_15 : UInt = bit-or(a, b)
    orout := T_15
    node T_16 : UInt = bit-xor(a, b)
    xorout := T_16