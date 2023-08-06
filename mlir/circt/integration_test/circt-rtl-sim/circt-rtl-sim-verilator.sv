// REQUIRES: verilator
// RUN: circt-rtl-sim.py --cycles 2 %s | FileCheck %s

module top(
  input clk,
  input rst
);

  always@(posedge clk)
    if (~rst)
      $display("tock");
  // CHECK:      tock
  // CHECK-NEXT: tock

endmodule
