
module AnalogReaderBlackBox(
  inout [31:0] bus,
  output [31:0] out
);
  assign bus = 32'dz;
  assign out = bus;
endmodule

module AnalogWriterBlackBox(
  inout [31:0] bus,
  input [31:0] in
);
  assign bus = in;
endmodule

module AnalogBlackBox #(
  parameter index=0
) (
  inout [31:0] bus,
  input port_0_in_valid,
  input [31:0] port_0_in_bits,
  output [31:0] port_0_out
);
  assign port_0_out = bus;
  assign bus = (port_0_in_valid)? port_0_in_bits + index : 32'dZ;
endmodule
