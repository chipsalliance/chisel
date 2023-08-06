module adder (
    input [15:0] in1, in2,
    output logic [15:0] out
);
    always_comb begin
        out = in1 + in2;
    end
endmodule
