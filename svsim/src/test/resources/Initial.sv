// SPDX-License-Identifier: Apache-2.0

module Initial(output b);

  reg a = 1'b0;

  assign b = a;

  initial
    a = 1'b1;

endmodule
