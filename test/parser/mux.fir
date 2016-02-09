; RUN: firrtl -i %s -o %s.out -X firrtl && cat %s.out | FileCheck %s
circuit GCD : 
  module GCD : 
    input e : UInt<1>
    input f : UInt<1>
    input g : UInt<1>
    output o : UInt<1>
    node y = f
    node z = f
    node x = mux(UInt<1>(0),y,z)
    o <= x

    

