; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      input value : UInt<32>
      input in : {x : UInt<32>, y : UInt<32>}
      wire m :{x : UInt<32>, y : UInt<32>}[2][2] 
      wire i : UInt

      m[0][0] := in
      m[1][0] := in
      m[0][1] := in
      m[1][1] := in
      i := UInt("h1")

      write accessor a = m[i] 
      write accessor b = a[i] 
      b.x := value

;CHECK: a := m$0
;CHECK: when eqv(i_1, UInt("h1")) : a := m$1



;CHECK: Done!
