; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      input in : UInt<32>
      input i : UInt<1>
      wire m : UInt<32>[2]
      m[0] := UInt("h1")
      m[1] := UInt("h1")
      infer accessor a = m[i] 
      a := in

;CHECK: wire a_2 : UInt<32>
;CHECK: when eqv(i_1, UInt("h0")) : m$0 := a_2
;CHECK: when eqv(i_1, UInt("h1")) : m$1 := a_2



;CHECK: Done!
