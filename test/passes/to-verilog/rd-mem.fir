; RUN: firrtl -i %s -o %s.v -X verilog -p c &> %s.out ; cat %s.v | FileCheck %s

circuit top :
   module top :
      output rdata : UInt<32>
      input index : UInt<2>
      input ren : UInt<1>
      input clk : Clock

      mem m : 
         data-type => UInt<32>
         depth => 4
         read-latency => 1
         write-latency => 1
         reader => c
      m.c.addr <= index
      m.c.en <= ren
      m.c.clk <= clk
      rdata <= m.c.data

; CHECK: module top(
; CHECK:    output [31:0] rdata,
; CHECK:    input  [1:0] index,
; CHECK:    input   ren,
; CHECK:    input   clk,
; CHECK: );
; CHECK:    reg [31:0] m [0:3];
; CHECK:    wire [31:0] m_c_data;
; CHECK:    wire [1:0] m_c_addr;
; CHECK:    wire  m_c_en;
; CHECK:    wire  m_c_clk;
; CHECK:    reg [1:0] GEN;
; CHECK:    reg  GEN_1;
; CHECK:    assign rdata = m_c_data
; CHECK:    assign m_c_addr = index
; CHECK:    assign m_c_en = ren
; CHECK:    assign m_c_clk = clk
; CHECK: `ifndef SYNTHESIS
; CHECK:   integer initvar;
; CHECK:   initial begin
; CHECK:     #0.002;
; CHECK:    for (initvar = 0; initvar < 4; initvar = initvar+1)
; CHECK:       m[initvar] = {1{$random}};
; CHECK:   end
; CHECK: `endif
; CHECK:    always @(posedge m_c_clk) begin
; CHECK:       GEN <= m_c_addr
; CHECK:       GEN_1 <= m_c_en
; CHECK:       if(GEN_1) begin
; CHECK:          m_c_data <= m[GEN]
; CHECK:       end
; CHECK:    end
; CHECK: endmodule
