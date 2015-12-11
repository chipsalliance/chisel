module BlackBoxInverter(
    input  [0:0] clk,
    input  [0:0] reset,
    output [0:0] io_in,
    output [0:0] io_out
);
  assign io_out = !io_in;
endmodule

module BlackBoxPassthrough(
    input  [0:0] clk,
    input  [0:0] reset,
    output [0:0] io_in,
    output [0:0] io_out
);
  assign io_out = io_in;
endmodule
