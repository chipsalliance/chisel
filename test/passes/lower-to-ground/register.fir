; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower Types
         circuit top :
            module top :
               input a : UInt<16>
               input b : UInt<16>
               input clk : Clock
               input reset : UInt<1>

               wire q : { x : UInt, y : SInt }
               q.x <= UInt(0)
               q.y <= SInt(0)
               reg r1 : { x : UInt, y : SInt },clk,reset,q

      ; CHECK: wire q{{[_$]+}}x : UInt
      ; CHECK: wire q{{[_$]+}}y : SInt
      ; CHECK: reg r1{{[_$]+}}x : UInt<1>, clk, reset, q_x
      ; CHECK: reg r1{{[_$]+}}y : SInt<1>, clk, reset, q_y

; CHECK: Finished Lower Types
