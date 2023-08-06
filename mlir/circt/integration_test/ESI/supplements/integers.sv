// Auxiliary file for tests in this directory. Contains hand-coded modules for
// use in the ESI systems under test.

// Produce a stream of incrementing integers.
module IntCountProd (
  input clk,
  input rst,
  IValidReady_i32.sink ints
);
  logic unsigned [31:0] count;
  assign ints.valid = ~rst;
  assign ints.data = count;

  always@(posedge clk) begin
    if (rst)
      count <= 32'h0;
    else if (ints.ready)
      count <= count + 1'h1;
  end
endmodule

// Accumulate a stream of integers. Randomly backpressure. Print status every
// cycle. Output the total over a raw port.
module IntAcc (
  input clk,
  input rst,
  IValidReady_i32.source ints,
  output unsigned [31:0] totalOut
);
  logic unsigned [31:0] total;

  // De-assert ready randomly
  int unsigned randReady;
  assign ints.ready = ~rst && (randReady > 25);

  always@(posedge clk) begin
    randReady <= $urandom_range(100, 0);
    if (rst) begin
      total <= 32'h0;
    end else begin
      $display("Total: %10d", total);
      $display("Data: %5d", ints.data);
      if (ints.valid && ints.ready)
        total <= total + ints.data;
    end
  end
endmodule

// Accumulate a stream of integers. Print status every cycle. Output the total
// over an ESI channel.
module IntAccNoBP (
  input clk,
  input rst,
  IValidReady_i32.source ints,
  IValidReady_i32.sink totalOut
);
  logic unsigned [31:0] total;
  assign totalOut.data = total;

  always@(posedge clk) begin
    if (rst) begin
      total <= 32'h0;
      ints.ready <= 1;
      totalOut.valid <= 0;
    end else begin
      if (ints.valid && ints.ready) begin
        total <= total + ints.data;
        totalOut.valid <= 1;
        ints.ready <= totalOut.ready;
        $display("Total: %10d", total);
        $display("Data: %5d", ints.data);
      end else if (totalOut.valid && totalOut.ready) begin
        totalOut.valid <= 0;
        ints.ready <= 1;
      end
    end
  end
endmodule

module IntArrSum (
  input clk,
  input rst,
  IValidReady_ArrayOf4xsi13.source arr,
  IValidReady_ArrayOf2xui24.sink totalOut
);

  assign totalOut.valid = arr.valid;
  assign arr.ready = totalOut.ready;
  assign totalOut.data[0] = 24'($signed(arr.data[0])) + 24'($signed(arr.data[1]));
  assign totalOut.data[1] = 24'($signed(arr.data[2])) + 24'($signed(arr.data[3]));
endmodule

module Encryptor (
  input clk,
  input rst,
  IValidReady_Struct.source in,
  IValidReady_Struct1.source cfg,
  IValidReady_Struct.sink x
);
  logic [255:0] otp;
  logic encrypt;
  logic otpValid;

  assign x.data.blob = encrypt ? (in.data.blob ^ otp) : in.data.blob;

  assign x.valid = otpValid && in.valid;
  assign in.ready = x.ready && otpValid;
  assign x.data.encrypted = encrypt ? ~in.data.encrypted : in.data.encrypted;

  always@(posedge clk) begin
    if (cfg.valid) begin
      otp <= cfg.data.otp;
      encrypt <= cfg.data.encrypt;
      otpValid <= 1;
    end
    if (rst)
      otpValid <= 0;
  end
endmodule
