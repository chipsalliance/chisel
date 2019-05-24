// See LICENSE for license details.
module LargeParam #(parameter DATA=0, WIDTH=1) (
  output [WIDTH-1:0] out
);
  assign out = DATA;
endmodule

