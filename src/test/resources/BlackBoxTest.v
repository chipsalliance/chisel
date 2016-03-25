module BlackBoxInverter(
    input  [0:0] clk,
    input  [0:0] reset,
    output [0:0] io$in,
    output [0:0] io$out
);
  assign io$out = !io$in;
endmodule

module BlackBoxPassthrough(
    input  [0:0] clk,
    input  [0:0] reset,
    output [0:0] io$in,
    output [0:0] io$out
);
  assign io$out = io$in;
endmodule
