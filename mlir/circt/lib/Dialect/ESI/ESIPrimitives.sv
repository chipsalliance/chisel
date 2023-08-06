//===- ESIPrimitives.sv - Primitive RTL modules for ESI -----*- verilog -*-===//
//
// Some ESI ops lower into external modules -- primitives -- which live here.
//
//===----------------------------------------------------------------------===//

/// ESI_PipelineStage: the primitive which ESI uses to buffer data, like a relay
/// station. This implementation is double buffered locally and the backpressure
/// (x_ready) is fully pipelined. There is no combinational loop between a_valid
/// and a_ready. Will not introduce pipeline bubbles. (And won't drop tokens.)
///
/// QoR -- Setup: virtual pin project with N of these modules daisy-chained
/// targetting 800 MHz. VPP results are known to be unrealisitic since they have
/// no routing or placement pressure. There is also a lot of placement/routing
/// seed variation in the timing, so round numbers are presented here. The Agilex
/// timing models are preliminary. These are just to show area efficiency and
/// demonstrate the speed potential.
///
/// | N   | WIDTH | Device    | FMax    | ALMs   | ALMs / stage / bit |
/// | --- | ----- | --------- | ------- | -----  | ------------------ |
/// | 64  | 128   | StratixV  | 475 MHz |  8,658 | 1.06               |
/// | 64  | 16    | StratixV  | 550 MHz |  1,362 | 1.33               |
/// | 32  | 16    | StratixV  | 580 MHz |    690 | 1.35               |
/// | --- | ----- | --------- | ------- | ------ | ------------------ |
/// | 64  | 128   | Arria10   | 625 MHz |  8,734 | 1.07               |
/// | 64  | 16    | Arria10   | 640 MHz |  1,379 | 1.34               |
/// | 32  | 16    | Arria10   | 645 MHz |    697 | 1.35               |
/// | --- | ----- | --------- | ------- | ------ | ------------------ |
/// | 64  | 128   | Stratix10 | 615 MHz | 10,733 | 1.31               |
/// | 64  | 16    | Stratix10 | 700 MHz |  1,625 | 1.58               |
/// | 32  | 16    | Stratix10 | 730 MHz |    858 | 1.68               |
/// | --- | ----- | --------- | ------- | ------ | ------------------ |
/// | 64  | 128   | Agilex    | 750 MHz |  9,344 | 1.14               |
/// | 64  | 16    | Agilex    | 900 MHz |  1,468 | 1.43               |
/// | 32  | 16    | Agilex    | 950 MHz |    845 | 1.65               |
/// | --- | ----- | --------- | ------- | ------ | ------------------ |
///
module ESI_PipelineStage # (
  int WIDTH = 8
) (
  input logic clk,
  input logic rst,

  // Input LI channel.
  input logic a_valid,
  input logic [WIDTH-1:0] a,
  output logic a_ready,

  // Output LI channel.
  output logic x_valid,
  output logic [WIDTH-1:0] x,
  input logic x_ready
);

  // Output registers.
  logic [WIDTH-1:0] x_reg;
  logic x_valid_reg;
  logic x_ready_reg;
  assign x = x_reg;
  assign x_valid = x_valid_reg;

  // Register the backpressure.
  always_ff @(posedge clk)
    x_ready_reg <= rst ? 1'b0 : x_ready;

  // We are transmitting a token on this cycle.
  wire xmit = x_valid && x_ready;

  // Lookaside buffer. If the output register is occupied, put the incoming
  // token here instead.
  logic [WIDTH-1:0] l;
  logic l_valid;

  // We're only ready to take a token if we're gonna have space. We have space
  // if the lookaside reg is unoccupied. We'll have space if the output could
  // have accepted a token last cycle, which implies that the lookaside (if
  // occupied) is shifting this cycle.
  assign a_ready = ~l_valid || x_ready_reg;

  // Did we accept a token this cycle?
  wire a_rcv = a_ready && a_valid;

  always_ff @(posedge clk) begin
    if (rst) begin
      l_valid <= 1'b0;
      x_valid_reg <= 1'b0;
    end else begin
      // If we have an empty output reg due to a transmit and the lookaside is
      // also empty, load the input into the output reg.
      if (xmit && !l_valid) begin
        x_reg <= a;
        x_valid_reg <= a_rcv;

      // If we have an empty output reg due to a transmit and the lookaside is
      // full, load the lookaside into the output reg and put the input in the
      // lookaside buffer.
      end if (xmit && l_valid) begin
        x_reg <= l;
        x_valid_reg <= 1'b1;
        l <= a;
        l_valid <= a_rcv;

      // If we didn't transmit but we did accept a token:
      end else if (~xmit && a_rcv) begin
        // Make sure the lookaside has space (since if it's full, we shouldn't
        // have gotten a token).
        assert (~l_valid);
        // If the output reg is occupied, place it in the lookaside reg.
        if (x_valid_reg) begin
          l <= a;
          l_valid <= 1'b1;
        // If the output reg is empty, put the input there.
        end else begin
          x_reg <= a;
          x_valid_reg <= 1'b1;
        end
      end
    end
  end
endmodule
