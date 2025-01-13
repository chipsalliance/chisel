// SPDX-License-Identifier: Apache-2.0

module Initial(input clock);

  reg a = 1'b0;

  initial
    a = 1'b1;

  always @(posedge clock) begin
    assert(a == 1'b1);
  end

endmodule
