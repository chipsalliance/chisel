; RUN: firrtl -i %s -o %s.v -X verilog ; cat %s.v | FileCheck %s

;CHECK: module Bug(
;CHECK:    input   clk,
;CHECK:    input   a,
;CHECK:    input   b
;CHECK: );
;CHECK:    wire  GEN_0;
;CHECK:    assign GEN_0 = a & b;
;CHECK:    always @(posedge clk) begin
;CHECK:       `ifndef SYNTHESIS
;CHECK:          if(1'h1) begin
;CHECK:             $fwrite(32'h80000002,"%d\n",GEN_0);
;CHECK:          end
;CHECK:       `endif
;CHECK:    end
;CHECK: endmodule

circuit Bug :
  module Bug :
    input clk : Clock
    input a : UInt<1>
    input b : UInt<1>

    printf(clk, UInt<1>(1), "%d\n", and(a, b))
