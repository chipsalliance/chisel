// SPDX-License-Identifier: Apache-2.0
module AdderExtModule(
  input [15:0] foo,
  output [15:0] bar
);
  assign bar = foo + 1;
endmodule

