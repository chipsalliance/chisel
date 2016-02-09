; RUN: firrtl -i %s -o %s.v -X verilog ; cat %s.v | FileCheck %s

;CHECK: module top(
;CHECK:    input   clk
;CHECK: );
;CHECK:    always @(posedge clk) begin
;CHECK:       `ifndef SYNTHESIS
;CHECK:          if(1'h1) begin
;CHECK:             $fwrite(32'h80000002,"This has an escaped quote (\") in it");
;CHECK:          end
;CHECK:       `endif
;CHECK:    end
;CHECK: endmodule

circuit top :
   module top :
      input clk : Clock
      printf(clk, UInt<1>(1), "This has an escaped quote (\") in it")
