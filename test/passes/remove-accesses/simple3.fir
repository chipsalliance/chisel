; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Remove Accesses
circuit top :
   module top :
      input in : UInt<32>
      input i : UInt<1>
      wire m : UInt<32>[2]
      m[0] <= UInt("h1")
      m[1] <= UInt("h1")
      wire a : UInt<32>
      m[i] <= a
      a <= in

;CHECK: wire GEN_0 : UInt<32>
;CHECK: when eq(UInt<1>("h0"), i) : m[0] <= GEN_0
;CHECK: when eq(UInt<1>("h1"), i) : m[1] <= GEN_0
;CHECK: GEN_0 <= a

;CHECK: Finished Remove Accesses



;CHECK: Done!
