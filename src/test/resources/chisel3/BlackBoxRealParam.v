module BlackBoxRealParam #(
  parameter real REAL = 0.0
) (
  output [63:0] out
);
  assign out = $realtobits(REAL);
endmodule
