module BlackBoxTypeParam #(
  parameter type T = bit
) (
  output T out
);
  assign out = {32'hdeadbeef}[$bits(out)-1:0];
endmodule
