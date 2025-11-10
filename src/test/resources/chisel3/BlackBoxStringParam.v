module BlackBoxStringParam #(
  parameter string STRING = "zero"
) (
  output [31:0] out
);
  assign out = (STRING == "one" )? 1 :
               (STRING == "two" )? 2 : 0;
endmodule
