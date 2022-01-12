<!Doctype html>
<html>
<title> Two side column </title>
<body>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <table border ="0">
        <h1>Creating a Module</h1>
        <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
    module foo (
                    input  a,
                    output b
                )
                assign b = a;
            endmodule
```

</td>
    <td>

```scala mdoc
    class Foo extends Module {
    val a = Input(Bool())
    val b = Output(Bool())
    b := a
    }
```

</td>
    <td>Text comes here</td>
         </tr>
    <tr>
<td>
Text comes here
</td>
<td>

```scala mdoc:invisible
import Chisel.Queue
import chisel3._
import chisel3.util.DecoupledIO
```

```scala mdoc:silent
class PassthroughGenerator(width: Int) extends Module {
val io = IO(new Bundle {
val in = Input(UInt(width.W))
val out = Output(UInt(width.W))
})
io.out := io.in
}
println(getVerilogString(new PassthroughGenerator(10)))
println(getVerilogString(new PassthroughGenerator(20)))
```
</td>
<td>

```
module PassthroughGenerator(
  input        clock,
  input        reset,
  input  [9:0] io_in,
  output [9:0] io_out
);
  assign io_out = io_in; // @[main.scala 13:10]
endmodule

module PassthroughGenerator(
input         clock,
input         reset,
input  [19:0] io_in,
output [19:0] io_out
);
assign io_out = io_in; // @[main.scala 13:10]
endmodule
```
</td>
         </tr>
    </table>
</body>
</html>

# Parameterizing a Module

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
module ParameterizedWidthAdder(

input [in0Width-1:0] in0,
input [in1Width-1:0] in1,
output [sumWidth-1:0] sum
);
parameter in0Width = 8;
parameter in1Width = 1;
parameter sumWidth = 9;

assign sum = in0 + in1;

endmodule
```

</td>
<td>

```
class ParameterizedWidthAdder(in0Width: Int, in1Width: Int, sumWidth: Int) extends Module {
  require(in0Width >= 0)
  require(in1Width >= 0)
  require(sumWidth >= 0)
  val io = IO(new Bundle {
    val in0 = Input(UInt(in0Width.W))
    val in1 = Input(UInt(in1Width.W))
    val sum = Output(UInt(sumWidth.W))
  })
  // a +& b includes the carry, a + b does not
  io.sum := io.in0 +& io.in1
}
```
</td>
             <td>Text comes here</td>
</tr>
<tr>
<td>

```
module TestBench;
wire [31:0] sum;
ParameterizedWidthAdder  #(32, 32, 32) my32BitAdderWithTruncation (32'b0, 32'b0, sum);
endmodule
```
</td>
            <td>

```
val my32BitAdderWithTruncation = Module(new ParameterizedWidthAdder(32, 32, 32)
```
</td>
             <td>Text comes here</td>
         </tr>
 <tr>
<td>
Text comes here
</td>
<td>

```scala mdoc:silent
class MyModule extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(4.W))
    val out = Output(UInt(4.W))
  })

val two  = 1 + 1
println(two)
val utwo = 1.U + 1.U
println(utwo)

io.out := io.in
}
println(getVerilogString(new MyModule))
```
</td>
<td>

```
2
UInt<1>(OpResult in MyModule)
module MyModule(
  input        clock,
  input        reset,
  input  [3:0] io_in,
  output [3:0] io_out
);
  assign io_out = io_in; // @[main.scala 19:10]
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>

# Wire assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
wire [31:0] 
a = 32'd42; 
wire [31:0] 
b = 32'hbabecafe; 
wire [15:0] c; 
assign c = 16'b1;
```

</td>
<td>

```scala mdoc:silent


class MyWireAssignmentModule extends Module {
 val a = WireDefault(42.U(32.W))
 val b = WireDefault("hbabecafe".U(32.W))  
 val c = Wire(UInt(16.W)) 
  c := "b1".U;
}
 

println(getVerilogString(new MyWireAssignment))
```
</td>
<td>

```
module MyWireAssignmentModule(
  input   clock,
  input   reset
);
endmodule

```
</td>
         </tr>
    </table>
<html>
<body>

# Register assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
<td>

```scala mdoc:silent
class RegisterModule extends Module {
  val io = IO(new Bundle {
    val in  = Input(UInt(12.W))
    val out = Output(UInt(12.W))
  })

  val registerWithInit = RegInit(42.U(12.W))
    registerWithInit := registerWithInit - 1.U
    io.out := io.in
}

println(getVerilogString(new RegisterModule))
```
</td>
<td>

```
module RegisterModule(
  input         clock,
  input         reset,
  input  [11:0] io_in,
  output [11:0] io_out
);
  assign io_out = io_in; // @[main.scala 18:10]
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>

# Case statement

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
<td>

```scala mdoc:silent
cclass CaseStatementModule extends Module {
  
   val a, b, c= IO(Input(UInt(3.W)))
    val sel = IO(Input(UInt(2.W)))

  val out = IO(Output(UInt(3.W)))
    out := 0.U
  
  switch (sel) {
  is ("b00".U) {
    
   out := a
    
  }
  is ("b01".U) {
    
   out := b
  }
  is ("b10".U) {
    
   out := c
  }
}
  };

println(getVerilogString(new CaseStatementModule))
```
</td>
<td>

```
module CaseStatementModule(
  input        clock,
  input        reset,
  input  [2:0] a,
  input  [2:0] b,
  input  [2:0] c,
  input  [1:0] sel,
  output [2:0] out
);
  wire [2:0] _GEN_0 = 2'h2 == sel ? c : 3'h0; // @[main.scala 16:16 28:8 14:9]
  wire [2:0] _GEN_1 = 2'h1 == sel ? b : _GEN_0; // @[main.scala 16:16 24:8]
  assign out = 2'h0 == sel ? a : _GEN_1; // @[main.scala 16:16 19:8]
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>

# ChiselEnum

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
<td>

```scala mdoc:
import Chisel.Queue
import chisel3._
import chisel3.util.DecoupledIO
import chisel3.util.{switch,is}
import chisel3.experimental.ChiselEnum
```

```scala mdoc:silent
class CaseStatementModule extends Module {
  
  object AluMux1Sel extends ChiselEnum {
  val selectRS1, selectPC = Value
  }
import AluMux1Sel._
   val a, b = IO(Input(UInt(3.W)))
    val sel = IO(Input(AluMux1Sel()))
  val out = IO(Output(UInt(3.W)))
    out := 0.U
  switch (sel) {
  is (selectRS1) {
   out := a
  }
  is (selectPC) {
   out := b
  }
}
  };

println(getVerilogString(new CaseStatementModule))
```
</td>
<td>

```
module CaseStatementModule(
  input        clock,
  input        reset,
  input  [2:0] a,
  input  [2:0] b,
  input        sel,
  output [2:0] out
);
  wire [2:0] _GEN_0 = sel ? b : 3'h0; // @[main.scala 22:16 30:8 20:9]
  assign out = ~sel ? a : _GEN_0; // @[main.scala 22:16 25:8]
endmodule
```
</td>
         </tr>
  <tr>
<td>Text comes here</td>
<td>

```scala mdoc:silent
class CaseStatementModule extends Module {
  
  object AluMux1Sel extends ChiselEnum {
   val INIT  = Value(0x03.U) 
    val IDLE  = Value(0x13.U) 
    val START = Value(0x17.U) 
    val READY = Value(0x23.U) 
    
  }
   import AluMux1Sel._
  
    val state = RegInit(INIT)
  val nextState = IO(Output(AluMux1Sel()))
  nextState := state

  
  switch (state) {
  is (INIT) {
    
   state := IDLE 
    
  }
  is (IDLE) {
    
   state := START
  }
  is (START) {
    
   state := READY
  }
 is (READY) {
    
   state := IDLE
  }
  }
}
println(getVerilogString(new CaseStatementModule))
```
</td>
<td>

```
module CaseStatementModule(
  input        clock,
  input        reset,
  output [5:0] nextState
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
`endif // RANDOMIZE_REG_INIT
  reg [5:0] state; // @[main.scala 20:24]
  wire [5:0] _GEN_0 = 6'h23 == state ? 6'h13 : state; // @[main.scala 25:18 41:10 20:24]
  assign nextState = state; // @[main.scala 22:13]
  always @(posedge clock) begin
    if (reset) begin // @[main.scala 20:24]
      state <= 6'h3; // @[main.scala 20:24]
    end else if (6'h3 == state) begin // @[main.scala 25:18]
      state <= 6'h13; // @[main.scala 28:10]
    end else if (6'h13 == state) begin // @[main.scala 25:18]
      state <= 6'h17; // @[main.scala 33:10]
    end else if (6'h17 == state) begin // @[main.scala 25:18]
      state <= 6'h23; // @[main.scala 37:10]
    end else begin
      state <= _GEN_0;
    end
  end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
  integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
  `ifdef RANDOMIZE
    `ifdef INIT_RANDOM
      `INIT_RANDOM
    `endif
    `ifndef VERILATOR
      `ifdef RANDOMIZE_DELAY
        #`RANDOMIZE_DELAY begin end
      `else
        #0.002 begin end
      `endif
    `endif
`ifdef RANDOMIZE_REG_INIT
  _RAND_0 = {1{`RANDOM}};
  state = _RAND_0[5:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>

# SystemVerilog Interfaces

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>
<td>

```scala mdoc:silent
class MyModule extends Module {
val io = IO(new Bundle {
val in = Flipped(DecoupledIO(UInt(8.W)))
val out = DecoupledIO(UInt(8.W))
})

val tmp = Wire(DecoupledIO(UInt(8.W)))
tmp <> io.in
io.out <> tmp
io.out <> io.in
}
println(getVerilogString(new MyModule))
```
</td>
<td>

```
module MyModule(
  input        clock,
  input        reset,
  output       io_in_ready,
  input        io_in_valid,
  input  [7:0] io_in_bits,
  input        io_out_ready,
  output       io_out_valid,
  output [7:0] io_out_bits
);
  assign io_in_ready = io_out_ready; // @[main.scala 17:12]
  assign io_out_valid = io_in_valid; // @[main.scala 17:12]
  assign io_out_bits = io_in_bits; // @[main.scala 17:12]
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>

# Memory Modules

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>Text comes here</td>

<td>

```scala mdoc:silent
class ReadWriteSmem extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })
  val mem = SyncReadMem(1024, UInt(width.W))
  // Create one write port and one read port
  mem.write(io.addr, io.dataIn)
  io.dataOut := mem.read(io.addr, io.enable)
}
class ReadWriteMem extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })
  val mem = Mem(1024, UInt(width.W))
  // Create one write port and one read port
  mem.write(io.addr, io.dataIn)
  io.dataOut := mem.read(io.addr)
}
println(getVerilogString(new ReadWriteSmem))
println(getVerilogString(new ReadWriteMem))
```
</td>

<td>

```
module ReadWriteSmem(
input         clock,
input         reset,
input         io_enable,
input         io_write,
input  [9:0]  io_addr,
input  [31:0] io_dataIn,
output [31:0] io_dataOut
);
`ifdef RANDOMIZE_MEM_INIT
reg [31:0] _RAND_0;
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
reg [31:0] _RAND_1;
reg [31:0] _RAND_2;
`endif // RANDOMIZE_REG_INIT
reg [31:0] mem [0:1023]; // @[main.scala 15:24]
wire  mem_io_dataOut_MPORT_en; // @[main.scala 15:24]
wire [9:0] mem_io_dataOut_MPORT_addr; // @[main.scala 15:24]
wire [31:0] mem_io_dataOut_MPORT_data; // @[main.scala 15:24]
wire [31:0] mem_MPORT_data; // @[main.scala 15:24]
wire [9:0] mem_MPORT_addr; // @[main.scala 15:24]
wire  mem_MPORT_mask; // @[main.scala 15:24]
wire  mem_MPORT_en; // @[main.scala 15:24]
reg  mem_io_dataOut_MPORT_en_pipe_0;
reg [9:0] mem_io_dataOut_MPORT_addr_pipe_0;
assign mem_io_dataOut_MPORT_en = mem_io_dataOut_MPORT_en_pipe_0;
assign mem_io_dataOut_MPORT_addr = mem_io_dataOut_MPORT_addr_pipe_0;
assign mem_io_dataOut_MPORT_data = mem[mem_io_dataOut_MPORT_addr]; // @[main.scala 15:24]
assign mem_MPORT_data = io_dataIn;
assign mem_MPORT_addr = io_addr;
assign mem_MPORT_mask = 1'h1;
assign mem_MPORT_en = 1'h1;
assign io_dataOut = mem_io_dataOut_MPORT_data; // @[main.scala 18:14]
always @(posedge clock) begin
if (mem_MPORT_en & mem_MPORT_mask) begin
mem[mem_MPORT_addr] <= mem_MPORT_data; // @[main.scala 15:24]
end
mem_io_dataOut_MPORT_en_pipe_0 <= io_enable;
if (io_enable) begin
mem_io_dataOut_MPORT_addr_pipe_0 <= io_addr;
end
end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
`ifdef RANDOMIZE
`ifdef INIT_RANDOM
`INIT_RANDOM
`endif
`ifndef VERILATOR
`ifdef RANDOMIZE_DELAY
#`RANDOMIZE_DELAY begin end
`else
#0.002 begin end
`endif
`endif
`ifdef RANDOMIZE_MEM_INIT
_RAND_0 = {1{`RANDOM}};
for (initvar = 0; initvar < 1024; initvar = initvar+1)
mem[initvar] = _RAND_0[31:0];
`endif // RANDOMIZE_MEM_INIT
`ifdef RANDOMIZE_REG_INIT
_RAND_1 = {1{`RANDOM}};
mem_io_dataOut_MPORT_en_pipe_0 = _RAND_1[0:0];
_RAND_2 = {1{`RANDOM}};
mem_io_dataOut_MPORT_addr_pipe_0 = _RAND_2[9:0];
`endif // RANDOMIZE_REG_INIT
`endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
module ReadWriteMem(
input         clock,
input         reset,
input         io_enable,
input         io_write,
input  [9:0]  io_addr,
input  [31:0] io_dataIn,
output [31:0] io_dataOut
);
`ifdef RANDOMIZE_MEM_INIT
reg [31:0] _RAND_0;
`endif // RANDOMIZE_MEM_INIT
reg [31:0] mem [0:1023]; // @[main.scala 32:16]
wire  mem_io_dataOut_MPORT_en; // @[main.scala 32:16]
wire [9:0] mem_io_dataOut_MPORT_addr; // @[main.scala 32:16]
wire [31:0] mem_io_dataOut_MPORT_data; // @[main.scala 32:16]
wire [31:0] mem_MPORT_data; // @[main.scala 32:16]
wire [9:0] mem_MPORT_addr; // @[main.scala 32:16]
wire  mem_MPORT_mask; // @[main.scala 32:16]
wire  mem_MPORT_en; // @[main.scala 32:16]
assign mem_io_dataOut_MPORT_en = 1'h1;
assign mem_io_dataOut_MPORT_addr = io_addr;
assign mem_io_dataOut_MPORT_data = mem[mem_io_dataOut_MPORT_addr]; // @[main.scala 32:16]
assign mem_MPORT_data = io_dataIn;
assign mem_MPORT_addr = io_addr;
assign mem_MPORT_mask = 1'h1;
assign mem_MPORT_en = 1'h1;
assign io_dataOut = mem_io_dataOut_MPORT_data; // @[main.scala 35:14]
always @(posedge clock) begin
if (mem_MPORT_en & mem_MPORT_mask) begin
mem[mem_MPORT_addr] <= mem_MPORT_data; // @[main.scala 32:16]
end
end
// Register and memory initialization
`ifdef RANDOMIZE_GARBAGE_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_INVALID_ASSIGN
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_REG_INIT
`define RANDOMIZE
`endif
`ifdef RANDOMIZE_MEM_INIT
`define RANDOMIZE
`endif
`ifndef RANDOM
`define RANDOM $random
`endif
`ifdef RANDOMIZE_MEM_INIT
integer initvar;
`endif
`ifndef SYNTHESIS
`ifdef FIRRTL_BEFORE_INITIAL
`FIRRTL_BEFORE_INITIAL
`endif
initial begin
`ifdef RANDOMIZE
`ifdef INIT_RANDOM
`INIT_RANDOM
`endif
`ifndef VERILATOR
`ifdef RANDOMIZE_DELAY
#`RANDOMIZE_DELAY begin end
`else
#0.002 begin end
`endif
`endif
`ifdef RANDOMIZE_MEM_INIT
_RAND_0 = {1{`RANDOM}};
for (initvar = 0; initvar < 1024; initvar = initvar+1)
mem[initvar] = _RAND_0[31:0];
`endif // RANDOMIZE_MEM_INIT
`endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule
```
</td>
         </tr>
    </table>
<html>
<body>