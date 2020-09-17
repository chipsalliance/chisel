// SPDX-License-Identifier: Apache-2.0
module LargeParamUnsigned #(parameter DATA=0, WIDTH=1) (
  output [WIDTH-1:0] out
);
  assign out = DATA;
endmodule

module LargeParamSigned #(parameter DATA=0, WIDTH=1) (
  output signed [WIDTH-1:0] out
);
  assign out = DATA;
endmodule
