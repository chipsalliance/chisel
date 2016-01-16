; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Remove Accesses
circuit top :
   module top :
      input value : UInt<32>
      input in : {x : UInt<32>, y : UInt<32>}
      wire m :{x : UInt<32>, y : UInt<32>}[2][2] 
      wire i : UInt
      wire j : UInt

      m[0][0] <= in
      m[1][0] <= in
      m[0][1] <= in
      m[1][1] <= in
      i <= UInt("h1")
      j <= UInt("h1")

      m[i][j].x <= value

;CHECK: Done!
