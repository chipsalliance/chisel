// SPDX-License-Identifier: Apache-2.0

module SIntWire #(parameter WIDTH = 32)
(
  input [WIDTH-1:0] in,
  output [WIDTH-1:0] out
);
  assign out = in;
endmodule

// We are testing low-level APIs, no support for parameterized modules
module SIntTest
(
  input [7:0] in_8,
  output [7:0] out_8,
  input [30:0] in_31,
  output [30:0] out_31,
  input [31:0] in_32,
  output [31:0] out_32,
  input [32:0] in_33,
  output [32:0] out_33,
  output [7:0] out_const_8,
  output [30:0] out_const_31,
  output [31:0] out_const_32,
  output [32:0] out_const_33
);

SIntWire #(8) wire_8(.in(in_8), .out(out_8));
SIntWire #(31) wire_31(.in(in_31), .out(out_31));
SIntWire #(32) wire_32(.in(in_32), .out(out_32));
SIntWire #(33) wire_33(.in(in_33), .out(out_33));
assign out_const_8 = 8'h80;
assign out_const_31 = 31'b1000000000000000000000000000000;
assign out_const_32 = 32'b10000000000000000000000000000000;
assign out_const_33 = 33'b100000000000000000000000000000000;

endmodule
