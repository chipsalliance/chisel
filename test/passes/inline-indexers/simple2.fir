; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      output o1 : UInt
      output o2 : UInt
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] := UInt("h1")
      m[1] := UInt("h1")
      i := UInt("h1")
      infer accessor a = m[i] 
      o1 := a
      o2 := a

;CHECK: wire a_2 : UInt<32>
;CHECK: a_2 := m$0
;CHECK: when eqv(i_1, UInt("h1")) : a_2 := m$1
;CHECK: wire a_3 : UInt<32>
;CHECK: a_3 := m$0
;CHECK: when eqv(i_2, UInt("h1")) : a_3 := m$1



;CHECK: Done!
