; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Remove Accesses
circuit top :
   module top :
      output o : UInt
      o <= UInt(1)
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] <= UInt("h1")
      m[1] <= UInt("h1")
      i <= UInt("h1")
      when i :
         o <= m[i] 

;CHECK: when i :
;CHECK: GEN_0 <= m[0]
;CHECK: when eq(UInt("h1"), i) : GEN_0 <= m[1]
;CHECK: o <= GEN_0
;CHECK: Finished Remove Accesses
;CHECK: Done!
