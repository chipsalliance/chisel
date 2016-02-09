; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
circuit GCD : 
  module GCD : 
    input a : UInt<63>
    input b : UInt<63>
    input sign : UInt<1>
    output d : UInt
    ;wire T_205 : UInt
    node T_203 = UInt<6>("h3f")
    node normCount = not(T_203)
    node absIn = mux(sign, a, b)
    node T_205 = dshl(absIn, normCount)
    d <= T_205

