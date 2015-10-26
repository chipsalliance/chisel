; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

;CHECK: Inline Indexers
circuit top :
   module top :
      input in : {x : UInt<32>, y : UInt<32>}
      input i : UInt<1>
      wire m : {x : UInt<32>, y : UInt<32>}[2]
      m[0].x := UInt("h1")
      m[0].y := UInt("h1")
      m[1].x := UInt("h1")
      m[1].y := UInt("h1")
      infer accessor a = m[i] 
      a.x := in.x

;CHECK: wire a$x_2 : UInt<32>
;CHECK: node i_1 = i
;CHECK: when eqv(i_1, UInt("h0")) : m$0$x := a$x_2
;CHECK: when eqv(i_1, UInt("h1")) : m$1$x := a$x_2
;CHECK: a$x_2 := in$x
;CHECK: Finished Inline Indexers
;CHECK: Done!



