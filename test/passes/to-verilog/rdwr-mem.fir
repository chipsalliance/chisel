; RUN: firrtl -i %s -o %s.v -X verilog &> %s.out ; cat %s.v | FileCheck %s

circuit top :
   module top :
      output rdata : UInt<32>
      input wdata : UInt<32>
      input index : UInt<2>
      input ren : UInt<1>
      input wen : UInt<1>
      input clk : Clock

      smem m : UInt<32>[4]
      infer mport c = m[index],clk
      rdata is invalid
      when ren :
         rdata <= c
      when wen :
         c <= wdata


; CHECK: module top(
; CHECK:    output [31:0] rdata,
; CHECK:    input  [31:0] wdata,
; CHECK:    input  [1:0] index,
; CHECK:    input   ren,
; CHECK:    input   wen,
; CHECK:    input   clk
; CHECK: );
; CHECK:    reg [31:0] m [0:3];
; CHECK:    wire [31:0] m_c_rdata;
; CHECK:    wire [1:0] m_c_raddr;
; CHECK:    wire  m_c_ren;
; CHECK:    wire [31:0] m_c_wdata;
; CHECK:    wire [1:0] m_c_waddr;
; CHECK:    wire  m_c_wmask;
; CHECK:    wire  m_c_wen;
; CHECK:    wire  m_c_clk;
; CHECK:    reg [1:0] GEN_2;
; CHECK:    reg  GEN_3;
; CHECK:    reg [1:0] GEN_0;
; CHECK:    reg [31:0] GEN_1;
; CHECK:    assign rdata = m_c_rdata;
; CHECK:    assign m_c_clk = clk;
; CHECK:    assign m_c_raddr = index;
; CHECK:    assign m_c_ren = 1'h1;
; CHECK:    assign m_c_rdata = m[GEN_2];
; CHECK:    assign m_c_wdata = wen ? wdata : GEN_1;
; CHECK:    assign m_c_waddr = index;
; CHECK:    assign m_c_wmask = wen ? 1'h1 : 1'h0;
; CHECK:    assign m_c_wen = 1'h1;
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
; CHECK:       GEN_2 <= m_c_raddr;
; CHECK:       GEN_3 <= m_c_ren;
; CHECK:       if(m_c_wen & m_c_wmask) begin
; CHECK:          m[m_c_waddr] <= m_c_wdata;
; CHECK:       end
; CHECK:    end
; CHECK: endmodule
