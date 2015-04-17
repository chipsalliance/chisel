;RUN: firrtl %s abcefghipjklmno cw | tee %s.out | FileCheck %s
;CHECK: To Flo
circuit BitsOps : 
  module BitsOps : 
    input b : UInt(16)
    input a : UInt(16)
    output notout : UInt(16)
    output andout : UInt(16)
    output orout : UInt(16)
    output xorout : UInt(16)
    
    node T_13 = bit-not(a)
    notout := T_13
    node T_14 = bit-and(a, b)
    andout := T_14
    node T_15 = bit-or(a, b)
    orout := T_15
    node T_16 = bit-xor(a, b)
    xorout := T_16
;CHECK: Finished To Flo
