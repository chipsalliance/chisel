; RUN: firrtl -i %s -o %s.v -X verilog &> %s.out ; cat %s.v | FileCheck %s

circuit top :
   module top :
      input wdata : UInt<32>
      input index : UInt<2>
      input wen : UInt<1>
      input clk : Clock

      smem m : UInt<32>[4],clk
      write accessor c = m[index] 
      when wen :
         c <= wdata

; CHECK: module top(
; CHECK:     input  [31:0] wdata,
; CHECK:     input  [1:0] index,
; CHECK:     input  [0:0] wen,
; CHECK:     input  [0:0] clk
; CHECK: );
; CHECK:   reg  [31:0] m [0:3];
; CHECK: `ifndef SYNTHESIS
; CHECK:   integer initvar;
; CHECK:   initial begin
; CHECK:     #0.002;
; CHECK:     for (initvar = 0; initvar < 4; initvar = initvar+1)
; CHECK:       m[initvar] = {1{$random}};
; CHECK:   end
; CHECK: `endif
; CHECK:   always @(posedge clk) begin
; CHECK:     if(wen) begin
; CHECK:       m[index] <= wdata;
; CHECK:     end
; CHECK:   end
; CHECK: endmodule

