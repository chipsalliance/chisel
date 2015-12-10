; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Expand Accessors
circuit top :
   module top :
      output o1 : UInt
      output o2 : UInt
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] <= UInt("h1")
      m[1] <= UInt("h1")
      i <= UInt("h1")
      infer accessor a = m[i] ;CHECK: indexer a = (m$0 m$1)[i] : UInt<32>
      o1 <= a
      o2 <= a


