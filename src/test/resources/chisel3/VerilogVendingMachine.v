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

  assign dispense = (state == sOk) ? 1'd1 : 1'd0;

  always @(posedge clock) begin
    if (reset) begin
      state <= sIdle;
    end else begin
      case (state)
      sIdle: begin
        if (nickel) state <= s5;
        else if (dime) state <= s10;
        else state <= state;
      end
      s5: begin
        if (nickel) state <= s10;
        else if (dime) state <= s15;
        else state <= state;
      end
      s10: begin
        if (nickel) state <= s15;
        else if (dime) state <= sOk;
        else state <= state;
      end
      s15: begin
        if (nickel) state <= sOk;
        else if (dime) state <= sOk;
        else state <= state;
      end
      sOk: begin
        state <= sIdle;
      end
      endcase
    end
  end
endmodule
