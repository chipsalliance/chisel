// See LICENSE for license details.
module ParameterizedExtModule(
  input [15:0] foo,
  output [15:0] bar
);
  parameter VALUE = 0;
  parameter STRING = "one";
  parameter REAL = 1.0;
  parameter type TYP = bit;
  wire [15:0] fizz;
  wire [15:0] buzz;
  wire TYP tpe;
  assign bar = foo + VALUE + fizz + buzz + tpe;
  assign fizz = (STRING == "two")? 2 : (STRING == "one")? 1 : 0;
  assign buzz = (REAL > 2.5E50)? 2 : (REAL < 0.0)? 1 : 0;
  assign tpe = 2; // Will give 0 if bit, 2 for any larger width
endmodule

