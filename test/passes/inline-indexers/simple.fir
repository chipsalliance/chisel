; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      output o : UInt
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] := UInt("h1")
      m[1] := UInt("h1")
      i := UInt("h1")
      infer accessor a = m[i] 
      o := a

;CHECK: a_2 := m$0
;CHECK: when eqv(i_1, UInt("h1")) : a_2 := m$1



;CHECK: Done!
