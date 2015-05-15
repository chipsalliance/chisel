; RUN: firrtl -i %s -o %s.flo -X flo -p cTwd | tee %s.out | FileCheck %s
; CHECK: Done!

circuit Risc : 
  module Risc : 
    output out : UInt<32>
    output valid : UInt<1>
    input boot : UInt<1>
    input isWr : UInt<1>
    input wrAddr : UInt<8>
    input wrData : UInt<32>
    
    mem file : UInt<32>[256]
    mem code : UInt<32>[256]
    reg pc : UInt<8>
    on-reset pc := Pad(UInt<8>(0),?)
    accessor inst = code[pc]
    node op = bits(inst, 31, 24)
    node rci = bits(inst, 23, 16)
    node rai = bits(inst, 15, 8)
    node rbi = bits(inst, 7, 0)
    node T_51 = eq(Pad(rai,?), Pad(UInt<1>(0),?))
    accessor T_52 = file[rai]
    node ra = mux(Pad(T_51,?), Pad(UInt<1>(0),?), Pad(T_52,?))
    node T_53 = eq(Pad(rbi,?), Pad(UInt<1>(0),?))
    accessor T_54 = file[rbi]
    node rb = mux(Pad(T_53,?), Pad(UInt<1>(0),?), Pad(T_54,?))
    wire rc : UInt<32>
    node T_55 = UInt<1>(0)
    valid := Pad(T_55,?)
    out := Pad(UInt<1>(0),?)
    rc := Pad(UInt<1>(0),?)
    when isWr : 
      accessor T_56 = code[wrAddr]
      T_56 := Pad(wrData,?)
    else : when boot : pc := Pad(UInt<1>(0),?)
      else : 
        node T_57 = eq(Pad(UInt<1>(0),?), Pad(op,?))
        when T_57 : 
          node T_58 = add-wrap(Pad(ra,?), Pad(rb,?))
          rc := Pad(T_58,?)
        node T_59 = eq(Pad(UInt<1>(1),?), Pad(op,?))
        when T_59 : 
          node T_60 = shl(rai, 8)
          node T_61 = bit-or(Pad(T_60,?), Pad(rbi,?))
          rc := Pad(T_61,?)
        out := Pad(rc,?)
        node T_62 = eq(Pad(rci,?), Pad(UInt<8>(255),?))
        when T_62 : 
          node T_63 = UInt<1>(1)
          valid := Pad(T_63,?)
        else : 
          accessor T_64 = file[rci]
          T_64 := Pad(rc,?)
        node T_65 = add-wrap(Pad(pc,?), Pad(UInt<1>(1),?))
        pc := Pad(T_65,?)
