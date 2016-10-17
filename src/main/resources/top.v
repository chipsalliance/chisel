module vcs_testbench;
  reg clk = 1;
  reg rst = 1;

  always #`CLOCK_PERIOD clk = ~clk;

  `TOP_MODULE `TOP_MODULE(.clk(clk), .reset(rst));
 
  initial begin
    // reset 5 cycles
    rst = 1;
    repeat (5) begin
      #`CLOCK_PERIOD;
      #`CLOCK_PERIOD;
    end
    rst = 0;
  end
endmodule
