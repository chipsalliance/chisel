; RUN: firrtl -i %s -o %s.v -X verilog ; cat %s.v | FileCheck %s

circuit Top :
   module Top :
      input a : SInt<2>
      output c : UInt<2>
      c <= and(a,asSInt(UInt(2)))

; CHECK: module Top(
; CHECK:    input  [1:0] a,
; CHECK:    output [1:0] c
; CHECK: );
; CHECK:    wire [1:0] GEN_0;
; CHECK:    assign c = $signed(a) & $signed(GEN_0);
; CHECK:    assign GEN_0 = $signed(2'h2);
; CHECK: endmodule

