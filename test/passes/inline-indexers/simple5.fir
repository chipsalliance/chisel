; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      output o : UInt
      o <= UInt(1)
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] <= UInt("h1")
      m[1] <= UInt("h1")
      i <= UInt("h1")
      when i  :
         infer accessor a = m[i] 
         o <= a

;CHECK: when i :
;CHECK: a_2 <= m$0
;CHECK: when eqv(i_1, UInt("h1")) : a_2 <= m$1
;CHECK: Finished Inline Indexers
;CHECK: Done!
