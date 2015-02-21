; RUN: firrtl %s abc | tee %s.out | FileCheck %s

circuit top :
   module A :
      ;CHECK: input reset : UInt(1)
      input x : UInt(16)
      output y : UInt(16)
      inst b of B
      ;CHECK: inst:b.reset := port:reset
   module B :
      input reset : UInt(1)
      ;CHECK: input reset : UInt(1)
      input x : UInt(16)
      output y : UInt(16)
      inst c of C
      ;CHECK: inst:c.reset := port:reset
   module C :
      ;CHECK: input reset : UInt(1)
      input a : UInt(16)
      input b : UInt(16)
   module top :
      ;CHECK: input reset : UInt(1)
      input a : UInt(16)
      input b : UInt(16)
      output z : UInt
      inst a of A
      ;CHECK: inst:a.reset := port:reset
