---
layout: docs
title:  "Verilog-vs-Chisel"
section: "chisel3"
---

<!Doctype html>
<html>

# Verilog vs Chisel Side-By-Side

This page serves as a quick introduction to Chisel for those familiar with Verilog. It is by no means a comprehensive guide of everything Chisel can do. Feel free to file an issue with suggestions of things you'd like to see added to this page.

```scala mdoc:invisible
import chisel3._
import chisel3.util.{switch, is}
import circt.stage.ChiselStage
import chisel3.util.{Cat, Fill, DecoupledIO}
```

<body>
    <!-- This script is needed so that Markdown and HTML will render together, see link to Stack overflow -->
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <table border ="0">
        <h1>Creating a Module</h1>
        <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```verilog
module Foo (
  input  a,
  output b
);
  assign b = a;
endmodule
```

</td>
    <td>

```scala mdoc
class Foo extends Module {
  val a = IO(Input(Bool()))
  val b = IO(Output(Bool()))
  b := a
}
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
         </tr>

<tr>
<td>

```verilog
module PassthroughGenerator(
  input  [width-1:0] in,
  output [width-1:0] out
);

  parameter width = 8;

  assign out = in;
endmodule
```
</td>
<td>

```scala mdoc:silent
class PassthroughGenerator(width: Int = 8) extends Module {
    val in = IO(Input(UInt(width.W)))
    val out = IO(Output(UInt(width.W)))

    out := in
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new PassthroughGenerator(10))
```
</td>
         </tr>
         <tr>
<td>

```verilog
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

```scala mdoc:silent
class ParameterizedWidthAdder(
  in0Width: Int,
  in1Width: Int,
  sumWidth: Int) extends Module {

  val in0 = IO(Input(UInt(in0Width.W)))
  val in1 = IO(Input(UInt(in1Width.W)))
  val sum = IO(Output(UInt(sumWidth.W)))

  // a +& b includes the carry, a + b does not
  sum := in0 +& in1
}
```
</td>
</tr>
    </table>
<html>
<body>

# Wire Assignment and Literal Values

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
<tr>
<td>

```verilog
module MyWireAssignmentModule ();

  wire [31:0] aa = 'd42;
  // Logical reg for use in always block, not real register
  reg [31:0] a;

  //
  always @(*) begin
    a = aa;
  end

  // Hex value initialization
  wire [31:0] b = 32'hbabecafe;

  // Declaration separate from Assignment
  wire [15:0] c;
  wire d;

  assign c = 16'b1;
  assign d = 1'b1;

  // Signed values
  wire signed [63:0] g;
  assign g = -’d5;

  wire signed [31:0] h = 'd5;

  reg signed[31:0] f;
  always@(*) begin
    f = ‘d5;
  end
endmodule
```


</td>
<td>

```scala mdoc:silent


class MyWireAssignmentModule extends Module {

    val aa = 42.U(32.W)
    val a = Wire(UInt(32.W))
    a := aa
    val b = "hbabecafe".U(32.W)
    val c = Wire(UInt(16.W))
    val d = Wire(Bool())
    c := "b1".U(16.W)
    d := true.B
    val g = Wire(SInt(64.W))
    g := -5.S
    val h = 5.asSInt(32.W)
    val f = Wire(SInt(32.W))
    f := 5.S
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new MyWireAssignmentModule)
```

</td>
</tr>
    </table>
<html>
<body>

# Register Declaration and Assignment

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
 <tr>
<td>

```verilog
module RegisterModule(
  input        clock,
  input        reset,
  input  [7:0] in,
  output [7:0] out,
  input        differentClock,
  input        differentSyncReset,
  input        differentAsyncReset
);

  reg [7:0] registerWithoutInit;
  reg [7:0] registerWithInit;
  reg [7:0] registerOnDifferentClockAndSyncReset;
  reg [7:0] registerOnDifferentClockAndAsyncReset;


  always @(posedge clock) begin
    registerWithoutInit <= in + 8'h1;
  end

  always @(posedge clock) begin
    if (reset) begin
      registerWithInit <= 8'd42;
    end else begin
      registerWithInit <= registerWithInit - 8'h1;
    end
  end

  always @(posedge differentClock) begin
    if (differentSyncReset) begin
      registerOnDifferentClockAndSyncReset <= 8'h42;
    end else begin
      registerOnDifferentClockAndSyncReset <= in - 8'h1;
    end
  end

  always @(posedge differentClock or posedge differentAsyncReset) begin
    if (differentAsyncReset) begin
      registerOnDifferentClockAndAsyncReset <= 8'h24;
    end else begin
      registerOnDifferentClockAndAsyncReset <= in + 8'h2;
    end
  end

  assign out = in +
    registerWithoutInit +
    registerWithInit +
    registerOnDifferentClockAndSyncReset +
    registerOnDifferentClockAndAsyncReset;
endmodule

```
</td>
<td>

```scala mdoc:silent
class RegisterModule extends Module {
  val in  = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  val differentClock = IO(Input(Clock()))
  val differentSyncReset = IO(Input(Bool()))

  val differentAsyncReset = IO(Input(AsyncReset()))

  val registerWithoutInit = Reg(UInt(8.W))

  val registerWithInit = RegInit(42.U(8.W))

  registerWithoutInit := in + 1.U

  registerWithInit := registerWithInit - 1.U

  val registerOnDifferentClockAndSyncReset = withClockAndReset(differentClock, differentSyncReset) {
    val reg = RegInit("h42".U(8.W))
    reg
  }
  registerOnDifferentClockAndSyncReset := in - 1.U

  val registerOnDifferentClockAndAsyncReset = withClockAndReset(differentClock, differentAsyncReset) {
    val reg = RegInit("h24".U(8.W))
    reg
  }
  registerOnDifferentClockAndAsyncReset := in + 2.U

  out := in +
    registerWithoutInit +
    registerWithInit +
    registerOnDifferentClockAndSyncReset +
    registerOnDifferentClockAndAsyncReset
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new RegisterModule)
```
</td>
         </tr>
    </table>
<html>
<body>

# Case Statements

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```verilog
module CaseStatementModule(
  input  [2:0] a,
  input  [2:0] b,
  input  [2:0] c,
  input  [1:0] sel,
  output reg [2:0] out
);

  always @(*)
    case (sel)
      2'b00: out <= a;
      2'b01: out <= b;
      2'b10: out <= c;
      default: out <= 3'b0;
    end
  end
endmodule
```
</td>
<td>

```scala mdoc:silent
class CaseStatementModule extends Module {
  val a, b, c= IO(Input(UInt(3.W)))
  val sel = IO(Input(UInt(2.W)))
  val out = IO(Output(UInt(3.W)))

  // default goes first
  out := 0.U

  switch (sel) {
    is ("b00".U) { out := a }
    is ("b01".U) { out := b }
    is ("b10".U) { out := c }
  }
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new CaseStatementModule)
```
</td>
         </tr>
    </table>
<html>
<body>

# Case Statements Using Enums

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```verilog
module CaseStatementEnumModule1 (
  input [2:0] rs1,
  input [2:0] pc,
  input AluMux1Sel sel,
  output reg [2:0] out);

  typedef enum {SELECT_RS1, SELECT_PC} AluMux1Sel;

  case(sel)
    SELECT_RS1: out <= rs1;
    SELECT_PC: out <= pc;
    default: out <= 3'b0;
  end
endmodule
```
</td>
<td>

```scala mdoc:silent
class CaseStatementEnumModule1 extends Module {

  object AluMux1Sel extends ChiselEnum {
    val selectRS1, selectPC = Value
  }

   import AluMux1Sel._
   val rs1, pc = IO(Input(UInt(3.W)))
   val sel = IO(Input(AluMux1Sel()))
   val out = IO(Output(UInt(3.W)))

   // default goes first
   out := 0.U

   switch (sel) {
     is (selectRS1) { out := rs1 }
     is (selectPC)  { out := pc }
  }
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new CaseStatementEnumModule1)
```
</td>
         </tr>
  <tr>
<td>

```verilog
module CaseStatementEnumModule2 (input clk);

  typedef enum {
    INIT = 3,
    IDLE = 'h13,
    START = 'h17,
    READY = 'h23 } StateValue;

  reg StateValue state;


  always @(posedge clk) begin
    case (state)
      IDLE    : state = START;
      START   : state = READY;
      READY   : state = IDLE ;
      default : state = IDLE ;
    endcase
  end
endmodule
```
</td>
<td>

```scala mdoc:silent
class CaseStatementEnumModule2 extends Module {

  object StateValue extends ChiselEnum {
   val INIT  = Value(0x03.U)
    val IDLE  = Value(0x13.U)
    val START = Value(0x17.U)
    val READY = Value(0x23.U)
  }
  import StateValue._
  val state = Reg(StateValue())


  switch (state) {
    is (INIT) {state := IDLE}
    is (IDLE) {state := START}
    is (START) {state := READY}
    is (READY) {state := IDLE}
  }
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new CaseStatementEnumModule2)
```
</td>
         </tr>
    </table>
<html>
<body>

<!--
# SystemVerilog Interfaces

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```
module MyInterfaceModule(
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
<td>

```scala mdoc:silent
class MyInterfaceModule extends Module {
val io = IO(new Bundle {
val in = Flipped(DecoupledIO(UInt(8.W)))
val out = DecoupledIO(UInt(8.W))
})

val tmp = Wire(DecoupledIO(UInt(8.W)))
tmp <> io.in
io.out <> tmp
io.out <> io.in
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new MyInterfaceModule)
```

</td>
         </tr>
    </table>

<html>
<body>
-->


# Multi-Dimensional Memories

<html>
<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```verilog
module ReadWriteSmem(
  input clock,
  input reset,
  input io_enable,
  input io_write,
  input [9:0] io_addr,
  input [31:0] io_dataIn,
  output [31:0] io_dataOut
);

reg [31:0] mem [0:1023];
reg [9:0] addr_delay;

assign io_dataOut = mem[addr_delay]

  always @(posedge clock) begin
    if (io_enable & io_write) begin
      mem[io_addr] <= io_data;
    end
    if (io_enable) begin
      addr_delay <= io_addr;
    end
  end
endmodule
```
</td>


<td>

```scala mdoc:silent
class ReadWriteSmem extends Module {
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(32.W))
    val dataOut = Output(UInt(32.W))
  })

  val mem = SyncReadMem(1024, UInt(32.W))

  // Create one write port and one read port
  mem.write(io.addr, io.dataIn)
  io.dataOut := mem.read(io.addr, io.enable)
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new ReadWriteSmem)
```
</td>
</tr>
<tr>
<td>

```verilog
module ReadWriteMem(
  input         clock,
  input         io_enable,
  input         io_write,
  input  [9:0]  io_addr,
  input  [31:0] io_dataIn,
  output [31:0] io_dataOut
  );

  reg [31:0] mem [0:1023];

  assign io_dataOut = mem[io_addr];

  always @(posedge clock) begin
    if (io_enable && io_write) begin
      mem[io_addr] <= io_dataIn;
    end
  end

endmodule
```

</td>

<td>

```scala mdoc:silent
class ReadWriteMem extends Module {
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(32.W))
    val dataOut = Output(UInt(32.W))
})
  val mem = Mem(1024, UInt(32.W))
  // Create one write port and one read port
  mem.write(io.addr, io.dataIn)
  io.dataOut := mem.read(io.addr)
}
```
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new ReadWriteMem)
```
</td>
</tr>
    </table>
<html>
<html>

# Operators

<body>
    <table border ="0">
          <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
          </tr>
         <tr>
<td>

```verilog
 module OperatorExampleModule(
  input         clock,
  input         reset,
  input  [31:0] x,
  input  [31:0] y,
  input  [31:0] c,
  output [31:0] add_res,
  output [31:0] sub_res,
  output [31:0] mod_res,
  output [31:0] div_res,
  output [31:0] and_res,
  output [31:0] or_res,
  output [31:0] xor_res,
  output [31:0] not_res,
  output [31:0] logical_not_res,
  output [31:0] mux_res,
  output [31:0] rshift_res,
  output [31:0] lshift_res,
  output        andR_res,
  output        logical_and_res,
  output        logical_or_res,
  output        equ_res,
  output        not_equ_res,
  output        orR_res,
  output        xorR_res,
  output        gt_res,
  output        lt_res,
  output        geq_res,
  output        leq_res,
  output        single_bitselect_res,
  output [63:0] mul_res,
  output [63:0] cat_res,
  output [1:0]  multiple_bitselect_res,
  output [95:0] fill_res
);
  assign add_res = x + y;
  assign sub_res = x - y;
  assign mod_res = x % y;
  assign mul_res = x * y;
  assign div_res = x / y;
  assign equ_res = x == y;
  assign not_equ_res = x != y;
  assign and_res = x & y;
  assign or_res = x | y;
  assign xor_res = x ^ y;
   assign not_res = ~x;
  assign logical_not_res = !(x == 32'h0);
   assign logical_and_res = x[0] && y[0];
  assign logical_or_res = x[0] || y[0];
  assign cat_res = {x,y};
  assign mux_res = c[0] ? x : y;
  assign rshift_res = x >> y[2:0];
  assign lshift_res = x << y[2:0];
  assign gt_res = x > y;
  assign lt_res = x < y;
  assign geq_res = x >= y;
  assign leq_res = x <= y;
  assign single_bitselect_res = x[1];
  assign multiple_bitselect_res = x[1:0];
  assign fill_res = {3{x}};
  assign andR_res = &x;
  assign orR_res = |x;
  assign xorR_res = ^x;





endmodule

```

</td>
<td>

```scala mdoc:silent
class OperatorExampleModule extends Module {

  val x, y, c = IO(Input(UInt(32.W)))

  val add_res, sub_res,
  mod_res, div_res, and_res,
  or_res, xor_res, not_res,
  logical_not_res, mux_res,
  rshift_res , lshift_res = IO(Output(UInt(32.W)))

  val logical_and_res, logical_or_res,
  equ_res, not_equ_res, andR_res,
  orR_res, xorR_res, gt_res,lt_res,
  geq_res, leq_res,single_bitselect_res = IO(Output(Bool()))

  val mul_res, cat_res= IO(Output(UInt(64.W)))

  val multiple_bitselect_res = IO(Output(UInt(2.W)))

  val fill_res = IO(Output(UInt((3*32).W)))

  add_res := x + y
  sub_res := x - y
  mod_res := x % y
  mul_res := x * y
  div_res := x / y
  equ_res := x === y
  not_equ_res := x =/= y
  and_res := x & y
  or_res := x | y
  xor_res := x ^ y
  not_res :=  ~x
  logical_not_res := !x
  logical_and_res := x(0) && y(0)
  logical_or_res := x(0) || y(0)
  cat_res := Cat(x, y)
  mux_res := Mux(c(0), x, y)
  rshift_res := x >> y(2, 0)
  lshift_res := x << y(2, 0)
  gt_res := x > y
  lt_res := x < y
  geq_res := x >= y
  leq_res := x <= y
  single_bitselect_res := x(1)
  multiple_bitselect_res := x(1, 0)
  fill_res:= Fill(3,x)
  andR_res := x.andR
  orR_res := x.orR
  xorR_res := x.xorR
}
```

```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new OperatorExampleModule)
```
</td>
         </tr>
    </table>
</body>
</html>
