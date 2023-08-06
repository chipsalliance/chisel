//===- Cosim_Endpoint.sv - ESI cosim primary RTL module -----*- verilog -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Package: Cosim_DpiPkg
//
// Main cosim <--> dpi bridge module.
//
//===----------------------------------------------------------------------===//

import Cosim_DpiPkg::*;

module Cosim_Endpoint
#(
  parameter string ENDPOINT_ID_EXT = "",
  parameter longint RECV_TYPE_ID = -1,
  parameter int RECV_TYPE_SIZE_BITS = -1,
  parameter longint SEND_TYPE_ID = -1,
  parameter int SEND_TYPE_SIZE_BITS = -1
)
(
  input  logic clk,
  input  logic rst,

  output logic DataOutValid,
  input  logic DataOutReady,
  output logic [RECV_TYPE_SIZE_BITS-1:0] DataOut,

  input  logic DataInValid,
  output logic DataInReady,
  input  logic [SEND_TYPE_SIZE_BITS-1:0] DataIn
);

  string ENDPOINT_ID_BASE = $sformatf("%m");
  string ENDPOINT_ID;
  if (ENDPOINT_ID_EXT != "")
    assign ENDPOINT_ID = $sformatf("%s.%s", ENDPOINT_ID_BASE, ENDPOINT_ID_EXT);
  else
    assign ENDPOINT_ID = ENDPOINT_ID_BASE;

  bit Initialized;

  // Handle initialization logic.
  always@(posedge clk) begin
    // We've been instructed to start AND we're uninitialized.
    if (!Initialized) begin
      int rc;
      rc = cosim_init();
      if (rc != 0)
        $error("Cosim init failed (%d)", rc);
      rc = cosim_ep_register(ENDPOINT_ID, SEND_TYPE_ID, SEND_TYPE_SIZE_BYTES,
                             RECV_TYPE_ID, RECV_TYPE_SIZE_BYTES);
      if (rc != 0)
        $error("Cosim endpoint (%d) register failed: %d", ENDPOINT_ID, rc);
      Initialized = 1'b1;
    end
  end

  /// *******************
  /// Data out management.
  ///

  localparam int RECV_TYPE_SIZE_BYTES = int'((RECV_TYPE_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int RECV_TYPE_SIZE_BITS_DIFF = RECV_TYPE_SIZE_BITS % 8;
  localparam int RECV_TYPE_SIZE_BYTES_FLOOR = int'(RECV_TYPE_SIZE_BITS/8);
  localparam int RECV_TYPE_SIZE_BYTES_FLOOR_IN_BITS
      = RECV_TYPE_SIZE_BYTES_FLOOR * 8;

  byte unsigned DataOutBuffer[RECV_TYPE_SIZE_BYTES-1:0];
  always @(posedge clk) begin
    if (~rst && Initialized) begin
      if (DataOutValid && DataOutReady) // A transfer occurred.
        DataOutValid <= 1'b0;

      if (!DataOutValid || DataOutReady) begin
        int data_limit;
        int rc;

        data_limit = RECV_TYPE_SIZE_BYTES;
        rc = cosim_ep_tryget(ENDPOINT_ID, DataOutBuffer, data_limit);
        if (rc < 0) begin
          $error("cosim_ep_tryget(%d, *, %d -> %d) returned an error (%d)",
            ENDPOINT_ID, RECV_TYPE_SIZE_BYTES, data_limit, rc);
        end else if (rc > 0) begin
          $error("cosim_ep_tryget(%d, *, %d -> %d) had data left over! (%d)",
            ENDPOINT_ID, RECV_TYPE_SIZE_BYTES, data_limit, rc);
        end else if (rc == 0) begin
          if (data_limit == RECV_TYPE_SIZE_BYTES)
            DataOutValid <= 1'b1;
          else if (data_limit == 0)
            begin end // No message.
          else
            $error(
              "cosim_ep_tryget(%d, *, %d -> %d) did not load entire buffer!",
              ENDPOINT_ID, RECV_TYPE_SIZE_BYTES, data_limit);
        end
      end
    end else begin
      DataOutValid <= 1'b0;
    end
  end

  // Assign packed output bit array from unpacked byte array.
  genvar iOut;
  generate
    for (iOut=0; iOut<RECV_TYPE_SIZE_BYTES_FLOOR; iOut++)
      assign DataOut[((iOut+1)*8)-1:iOut*8] = DataOutBuffer[iOut];
    if (RECV_TYPE_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, we've got to copy the extra bits.
      assign DataOut[(RECV_TYPE_SIZE_BYTES_FLOOR_IN_BITS +
                      RECV_TYPE_SIZE_BITS_DIFF - 1) :
                        RECV_TYPE_SIZE_BYTES_FLOOR_IN_BITS]
             = DataOutBuffer[RECV_TYPE_SIZE_BYTES - 1]
                            [RECV_TYPE_SIZE_BITS_DIFF - 1 : 0];
  endgenerate

  initial begin
    $display("RECV_TYPE_SIZE_BITS: %d", RECV_TYPE_SIZE_BITS);
    $display("RECV_TYPE_SIZE_BYTES: %d", RECV_TYPE_SIZE_BYTES);
    $display("RECV_TYPE_SIZE_BITS_DIFF: %d", RECV_TYPE_SIZE_BITS_DIFF);
    $display("RECV_TYPE_SIZE_BYTES_FLOOR: %d", RECV_TYPE_SIZE_BYTES_FLOOR);
    $display("RECV_TYPE_SIZE_BYTES_FLOOR_IN_BITS: %d",
             RECV_TYPE_SIZE_BYTES_FLOOR_IN_BITS);
  end


  /// **********************
  /// Data in management.
  ///

  localparam int SEND_TYPE_SIZE_BYTES = int'((SEND_TYPE_SIZE_BITS+7)/8);
  // The number of bits over a byte.
  localparam int SEND_TYPE_SIZE_BITS_DIFF = SEND_TYPE_SIZE_BITS % 8;
  localparam int SEND_TYPE_SIZE_BYTES_FLOOR = int'(SEND_TYPE_SIZE_BITS/8);
  localparam int SEND_TYPE_SIZE_BYTES_FLOOR_IN_BITS
      = SEND_TYPE_SIZE_BYTES_FLOOR * 8;

  assign DataInReady = 1'b1;
  byte unsigned DataInBuffer[SEND_TYPE_SIZE_BYTES-1:0];

  always@(posedge clk) begin
    if (~rst && Initialized) begin
      if (DataInValid) begin
        int rc;
        rc = cosim_ep_tryput(ENDPOINT_ID, DataInBuffer, SEND_TYPE_SIZE_BYTES);
        if (rc != 0)
          $error("cosim_ep_tryput(%d, *, %d) = %d Error! (Data lost)",
            ENDPOINT_ID, SEND_TYPE_SIZE_BYTES, rc);
      end
    end
  end

  // Assign packed input bit array to unpacked byte array.
  genvar iIn;
  generate
    for (iIn=0; iIn<SEND_TYPE_SIZE_BYTES_FLOOR; iIn++)
      assign DataInBuffer[iIn] = DataIn[((iIn+1)*8)-1:iIn*8];
    if (SEND_TYPE_SIZE_BITS_DIFF != 0)
      // If the type is not a multiple of 8, we've got to copy the extra bits.
      assign DataInBuffer[SEND_TYPE_SIZE_BYTES - 1]
                         [SEND_TYPE_SIZE_BITS_DIFF - 1:0] =
             DataIn[(SEND_TYPE_SIZE_BYTES_FLOOR_IN_BITS +
                     SEND_TYPE_SIZE_BITS_DIFF - 1) :
                       SEND_TYPE_SIZE_BYTES_FLOOR_IN_BITS];
  endgenerate

  initial begin
    $display("SEND_TYPE_SIZE_BITS: %d", SEND_TYPE_SIZE_BITS);
    $display("SEND_TYPE_SIZE_BYTES: %d", SEND_TYPE_SIZE_BYTES);
    $display("SEND_TYPE_SIZE_BITS_DIFF: %d", SEND_TYPE_SIZE_BITS_DIFF);
    $display("SEND_TYPE_SIZE_BYTES_FLOOR: %d", SEND_TYPE_SIZE_BYTES_FLOOR);
    $display("SEND_TYPE_SIZE_BYTES_FLOOR_IN_BITS: %d",
             SEND_TYPE_SIZE_BYTES_FLOOR_IN_BITS);
  end

endmodule
