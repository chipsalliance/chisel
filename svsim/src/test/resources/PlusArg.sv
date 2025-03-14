// SPDX-License-Identifier: Apache-2.0

module PlusArg(output reg value, output reg test);

  initial begin
    value = 0;
    test = 0;
    $value$plusargs("value=%d", value);
    if ($test$plusargs("test"))
      test = 1;
  end


endmodule
