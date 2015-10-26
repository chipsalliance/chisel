; RUN: firrtl -i %s -o %s.v -X verilog &> %s.out ; cat %s.v | FileCheck %s

circuit top :
   module top :
      output rdata : UInt<32>
      input wdata : UInt<32>
      input index : UInt<2>
      input ren : UInt<1>
      input wen : UInt<1>
      input clk : Clock

      smem m : UInt<32>[4],clk
      rdwr accessor c = m[index] 
      when ren :
         rdata := c
      when wen :
         c := wdata


; CHECK: module top(
; CHECK:     output [31:0] rdata,
; CHECK:     input  [1:0] index,
; CHECK:     input  [0:0] ren,
; CHECK:     input  [0:0] clk
; CHECK: );
; CHECK:   wire [31:0] c;
; CHECK:   reg  [31:0] m [0:3];
; CHECK:   reg  [1:0] index_1;
; CHECK: `ifndef SYNTHESIS
; CHECK:   integer initvar;
; CHECK:   initial begin
; CHECK:     #0.002;
; CHECK:     for (initvar = 0; initvar < 4; initvar = initvar+1)
; CHECK:       m[initvar] = {1{$random}};
; CHECK:     index_1 = {1{$random}};
; CHECK:   end
; CHECK: `endif
; CHECK:   assign rdata = m[index_1];
; CHECK:   always @(posedge clk) begin
; CHECK:     if(ren) begin
; CHECK:       index_1 <= index;
; CHECK:     end else if(wen) begin
; CHECK:       m[index] <= wdata;
; CHECK:     end
; CHECK:   end
; CHECK: endmodule

