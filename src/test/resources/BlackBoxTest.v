module BlackBoxInverter(
    input  [0:0] in,
    output [0:0] out
);
  assign out = !in;
endmodule

module BlackBoxPassthrough(
    input  [0:0] in,
    output [0:0] out
);
  assign out = in;
endmodule

module BlackBoxRegister(
    input  [0:0] clock,
    input  [0:0] in,
    output [0:0] out
);
  reg [0:0] register;
  always @(posedge clock) begin
    register <= in;
  end
  assign out = register;
endmodule

module BlackBoxConstant #(
  parameter int WIDTH=1,
  parameter int VALUE=1
) (
  output [WIDTH-1:0] out
);
  assign out = VALUE;
endmodule
