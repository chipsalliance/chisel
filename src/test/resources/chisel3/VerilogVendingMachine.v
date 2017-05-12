// See LICENSE for license details.

// A simple Verilog FSM vending machine implementation
module VerilogVendingMachine(
  input clock,
  input reset,
  input nickel,
  input dime,
  output dispense
);
  parameter sIdle = 3'd0, s5 = 3'd1, s10 = 3'd2, s15 = 3'd3, sOk = 3'd4;
  reg [2:0] state;
  wire [2:0] next_state;

  assign dispense = (state == sOk) ? 1'd1 : 1'd0;

  always @(*) begin
    case (state)
      sIdle: if (nickel) next_state <= s5;
             else if (dime) next_state <= s10;
             else next_state <= state;
      s5: if (nickel) next_state <= s10;
             else if (dime) next_state <= s15;
             else next_state <= state;
      s10: if (nickel) next_state <= s15;
             else if (dime) next_state <= sOk;
             else next_state <= state;
      s15: if (nickel) next_state <= sOk;
             else if (dime) next_state <= sOk;
             else next_state <= state;
      sOk: next_state <= sIdle;
    endcase
  end

  // Go to next state
  always @(posedge clock) begin
    if (reset) begin
      state <= sIdle;
    end else begin
      state <= next_state;
    end
  end
endmodule

