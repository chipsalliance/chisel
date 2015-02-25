; RUN: firrtl %s abc c | tee %s.out | FileCheck %s

; CHECK: Make Explicit Reset
circuit top :
   module A :
      ;CHECK: input reset : UInt(1)
      input x : UInt(16)
      output y : UInt(16)
      inst b of B
      ;CHECK: b.reset := reset
   module B :
      input reset : UInt(1)
      ;CHECK: input reset : UInt(1)
      input x : UInt(16)
      output y : UInt(16)
      inst c of C
      ;CHECK: c.reset := reset
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
      ;CHECK: a.reset := reset
;CHECK: Finished Make Explicit Reset
