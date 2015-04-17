; RUN: firrtl %s abcefghipj c | tee %s.out | FileCheck %s 
; CHECK: Expand Whens
circuit Risc : 
  module Risc : 
    output out : UInt(32)
    output valid : UInt(1)
    input boot : UInt(1)
    input isWr : UInt(1)
    input wrAddr : UInt(8)
    input wrData : UInt(32)
    
    mem file : UInt(32)[256]
    mem code : UInt(32)[256]
    node T_51 = UInt(0, 8)
    reg pc : UInt(8)
    pc.init := T_51
    node add_op = UInt(0, 1)
    node imm_op = UInt(1, 1)
    accessor inst = code[pc]
    node op = bits(inst, 31, 24)
    node rci = bits(inst, 23, 16)
    node rai = bits(inst, 15, 8)
    node rbi = bits(inst, 7, 0)
    node T_52 = UInt(0, 1)
    node T_53 = eq(rai, T_52)
    node T_54 = UInt(0, 1)
    accessor T_55 = file[rai]
    node ra = mux(T_53, T_54, T_55)
    node T_56 = UInt(0, 1)
    node T_57 = eq(rbi, T_56)
    node T_58 = UInt(0, 1)
    accessor T_59 = file[rbi]
    node rb = mux(T_57, T_58, T_59)
    wire rc : UInt(32)
    node T_60 = UInt(0, 1)
    valid := T_60
    node T_61 = UInt(0, 1)
    out := T_61
    node T_62 = UInt(0, 1)
    rc := T_62
    when isWr : 
      accessor T_63 = code[wrAddr]
      T_63 := wrData
    else : when boot : 
        node T_64 = UInt(0, 1)
        pc := T_64
      else : 
        node T_65 = eq(add_op, op)
        when T_65 : 
          node T_66 = add-wrap(ra, rb)
          rc := T_66
        node T_67 = eq(imm_op, op)
        when T_67 : 
          node T_68 = shl(rai, 8)
          node T_69 = bit-or(T_68, rbi)
          rc := T_69
        out := rc
        node T_70 = UInt(255, 8)
        node T_71 = eq(rci, T_70)
        when T_71 : 
          node T_72 = UInt(1, 1)
          valid := T_72
        else : 
          accessor T_73 = file[rci]
          T_73 := rc
        node T_74 = UInt(1, 1)
        node T_75 = add-wrap(pc, T_74)
        pc := T_75
; CHECK: Finished Expand Whens
