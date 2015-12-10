; RUN: firrtl -i %s -o %s.v -X verilog -p c 2>&1 | tee %s.out | FileCheck %s

; CHECK: Lower To Ground
         circuit top :
            module top :
               input a : UInt<16>
               input b : UInt<16>
               input clk : Clock
               input reset : UInt<1>
               output z : UInt

               reg r1 : { x : UInt, y : SInt },clk,reset
               wire q : { x : UInt, y : SInt }
               onreset r1 <= q    

      ; CHECK: reg r1{{[_$]+}}x : UInt
      ; CHECK: reg r1{{[_$]+}}y : SInt
      ; CHECK: wire q{{[_$]+}}x : UInt
      ; CHECK: wire q{{[_$]+}}y : SInt
      ; CHECK: onreset r1{{[_$]+}}x <= q{{[_$]+}}x
      ; CHECK: onreset r1{{[_$]+}}y <= q{{[_$]+}}y

; CHECK: Finished Lower To Ground
