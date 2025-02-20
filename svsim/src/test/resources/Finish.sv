module Finish(input clock);

  always @ (posedge clock)
    $finish;

endmodule
