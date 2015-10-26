; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      input value : UInt<32>
      input in : {x : UInt<32>, y : UInt<32>}
      wire m :{x : UInt<32>, y : UInt<32>}[2][2] 
      wire i : UInt
      wire j : UInt

      m[0][0] := in
      m[1][0] := in
      m[0][1] := in
      m[1][1] := in
      i := UInt("h1")
      j := UInt("h1")

      write accessor a = m[i] 
      write accessor b = a[j] 
      b.x := value

;CHECK: wire b$x_2 : UInt<32>
;CHECK: node j_1 = j
;CHECK: when eqv(j_1, UInt("h0")) :
;CHECK:    wire a$0$x_2 : UInt<32>
;CHECK:    node i_1 = i
;CHECK:    when eqv(i_1, UInt("h0")) :
;CHECK:       m$0$0$x := a$0$x_2         
;CHECK:    when eqv(i_1, UInt("h1")) :
;CHECK:       m$1$0$x := a$0$x_2         
;CHECK:    a$0$x_2 := b$x_2      
;CHECK: when eqv(j_1, UInt("h1")) :
;CHECK:    wire a$1$x_2 : UInt<32>
;CHECK:    node i_2 = i
;CHECK:    when eqv(i_2, UInt("h0")) :
;CHECK:       m$0$1$x := a$1$x_2         
;CHECK:    when eqv(i_2, UInt("h1")) :
;CHECK:       m$1$1$x := a$1$x_2         
;CHECK:    a$1$x_2 := b$x_2      
;CHECK: b$x_2 := value   



;CHECK: Done!
