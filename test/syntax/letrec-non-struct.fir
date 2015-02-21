; RUN: firrtl %s | tee %s.out | FileCheck %s
circuit top:
   module top:
      input x : UInt(16)
      output y : UInt(16)
      letrec:
         reg r : UInt(10)
      in:
         r := UInt(11)
