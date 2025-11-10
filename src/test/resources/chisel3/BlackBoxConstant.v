module BlackBoxConstant #(
  parameter int WIDTH=1,
  parameter int VALUE=1
) (
  output [WIDTH-1:0] out
);
  assign out = VALUE[WIDTH-1:0];
endmodule
