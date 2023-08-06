// REQUIRES: ieee-sim
// RUN: circt-rtl-sim.py --sim %ieee-sim --cycles 2 %s | FileCheck %s

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
