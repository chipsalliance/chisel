; RUN: firrtl -i %s -o %s.v -X verilog &> %s.out ; cat %s.v | FileCheck %s

circuit top :
   module top :
      input wdata : UInt<32>
      input index : UInt<2>
      input wen : UInt<1>
      input clk : Clock

      smem m : UInt<32>[4]
      write mport c = m[index],clk
      when wen :
         c <= wdata

; CHECK: module top(
; CHECK:    input  [31:0] wdata,
; CHECK:    input  [1:0] index,
; CHECK:    input   wen,
; CHECK:    input   clk
; CHECK: );
; CHECK:    reg [31:0] m [0:3];
; CHECK:    wire [31:0] m_c_data;
; CHECK:    wire [1:0] m_c_addr;
; CHECK:    wire  m_c_mask;
; CHECK:    wire  m_c_en;
; CHECK:    wire  m_c_clk;
; CHECK:    reg [1:0] GEN_0;
; CHECK:    reg [31:0] GEN_1;
; CHECK:    assign m_c_data = wen ? wdata : GEN_1;
; CHECK:    assign m_c_addr = index;
; CHECK:    assign m_c_mask = wen ? 1'h1 : 1'h0;
; CHECK:    assign m_c_en = 1'h1;
; CHECK:    assign m_c_clk = clk;
; CHECK: `ifndef SYNTHESIS
; CHECK:   integer initvar;
; CHECK:   initial begin
; CHECK:     #0.002;
; CHECK:    for (initvar = 0; initvar < 4; initvar = initvar+1)
; CHECK:       m[initvar] = {1{$random}};
; CHECK:    GEN_0 = {1{$random}};
; CHECK:    GEN_1 = {1{$random}};
; CHECK:   end
; CHECK: `endif
; CHECK:    always @(posedge m_c_clk) begin
; CHECK:       if(m_c_en & m_c_mask) begin
; CHECK:          m[m_c_addr] <= m_c_data;
; CHECK:       end
; CHECK:    end
; CHECK: endmodule

