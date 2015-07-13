; RUN: firrtl -i %s -o %s.flo -X flo -p c | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
         circuit top :
            module top :
               input a : UInt<16>
               input b : UInt<16>
               input clk : Clock
               input reset : UInt<1>
               output z : UInt

               reg r1 : { x : UInt, flip y : SInt },clk,reset
               wire q : { x : UInt, flip y : SInt }
               onreset r1 := q    

      ; CHECK: reg r1_x : UInt
      ; CHECK: reg r1_y : SInt
      ; CHECK: wire q_x : UInt
      ; CHECK: wire q_y : SInt
      ; CHECK: onreset r1_x := q_x
      ; CHECK: onreset q_y := r1_y   

; CHECK: Finished Lower To Ground
