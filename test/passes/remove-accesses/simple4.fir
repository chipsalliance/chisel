; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Remove Accesses
circuit top :
   module top :
      input in : {x : UInt<32>, y : UInt<32>}
      input i : UInt<1>
      wire m : {x : UInt<32>, y : UInt<32>}[2]
      m[0].x <= UInt("h1")
      m[0].y <= UInt("h1")
      m[1].x <= UInt("h1")
      m[1].y <= UInt("h1")
      m[i].x <= in.x

;CHECK: when eq(UInt<1>("h0"), i) : m[0].x <= GEN_0
;CHECK: when eq(UInt<1>("h1"), i) : m[1].x <= GEN_0
;CHECK: GEN_0 <= in
;CHECK: Finished Remove Accesses
;CHECK: Done!



