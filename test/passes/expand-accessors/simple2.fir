; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Done!
circuit top :
   module top :
      output o1 : UInt
      output o2 : UInt
      wire m : UInt<32>[2]
      wire i : UInt
      m[0] <= UInt("h1")
      m[1] <= UInt("h1")
      i <= UInt("h1")
      wire a : UInt<32>
      a <= m[i]
      o1 <= a
      o2 <= a


