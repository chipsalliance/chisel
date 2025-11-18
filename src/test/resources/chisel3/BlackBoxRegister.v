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
