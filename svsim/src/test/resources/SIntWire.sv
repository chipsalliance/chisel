// SPDX-License-Identifier: Apache-2.0

module SIntWire(
  input signed [31:0] in,
  output signed [31:0] out);

  assign out = in;//8'h80000000;
endmodule
