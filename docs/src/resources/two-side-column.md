<!Doctype html>
<html>
<title> Two side column </title>

```scala mdoc:invisible
import chisel3._
import chisel3.util.{switch, is}
import chisel3.stage.ChiselStage
import chisel3.experimental.ChiselEnum
import chisel3.util.{Cat, Fill, DecoupledIO}
```

<body>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <table border ="0">
        <h1>Creating a Module</h1>
        <tr>
            <td><b style="font-size:30px">Verilog</b></td>
            <td><b style="font-size:30px">Chisel</b></td>
         </tr>
         <tr>
<td>

```
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
  val a = Input(Bool())
  val b = Output(Bool())
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

```
module PassthroughGenerator(
  input        clock,
  input        reset,
  input  [width-1:0] io_in,
  output [width-1:0] io_out
);
 
  parameter width = 8;
  
  assign io_out = io_in;
endmodule
```
</td>
<td>

```scala mdoc:silent
class PassthroughGenerator(width: Int = 8) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(width.W))
    val out = Output(UInt(width.W))
  })
  io.out := io.in
}
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new PassthroughGenerator(10))
```
</td>
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
class ParameterizedWidthAdder(in0Width: Int, in1Width: Int, sumWidth: Int) extends Tester{
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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new MyWireAssignmentModule)
```
</td>
         </tr>
<tr>
<td>

```
module MyWireAssignmentModule(
  input   clock,
  input   reset
);
endmodule
```


</td>
<td>

```scala mdoc:silent
class MyWireAssignmentModule2 extends Module {
 val a = WireDefault(42.U(32.W))
 val aa = 42.U(32.W)
 val b = WireDefault("hbabecafe".U(32.W))  
 val c = Wire(UInt(16.W)) 
 val d = Wire(Bool())
 val e = Wire(UInt())
 val f = WireDefault("hdead".U)
 val g = Wire(SInt(64.W))
 val h = WireDefault(5.asSInt(32.W))
 val i = WireDefault((3.S(16.W)))

c := "b1".U;
d := true.B
e := "b1".U(3.W);
g := -5.S
}
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new MyWireAssignmentModule2)
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
         </tr>
         <tr>
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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new RegisterModule)
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
         </tr>
         <tr>
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

<td>

```scala mdoc:silent
class CaseStatementModule extends Module {
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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new CaseStatementModule)
```
</td>
         </tr>
    </table>
<html>
<body>

# Case statement Enum Module 

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
<td>


```scala mdoc:silent
class CaseStatementEnumModule1 extends Module {
  
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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new CaseStatementEnumModule1)
```
</td>
         </tr>
  <tr>
<td>

```
module CaseStatementEnumModule2 (input clk);
 
  typedef enum {INIT, IDLE, START, READY} t_state;
    t_state state;
    
    
 
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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new CaseStatementEnumModule2)
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
ChiselStage.emitVerilog(new MyInterfaceModule)
```

</td>
         </tr>
    </table>

<html>
<body>
-->


# Memory Modules

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

endmodule
```
</td>

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
```
```scala mdoc:invisible
ChiselStage.emitVerilog(new ReadWriteSmem)
ChiselStage.emitVerilog(new ReadWriteMem)
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
            <td><b style="font-size:30px">Generated Verilog</b></td>
         </tr>
         <tr>
<td>

```
 res10: String = """module OperatorExampleModule(
   input         clock,
   input         reset,
   input  [31:0] x,
   input  [31:0] y,
   input  [31:0] c,
   output [31:0] add_res,
   output [31:0] sub_res,
   output [31:0] per_res,
   output [31:0] mul_res,
   output [31:0] and_res,
   output [31:0] or_res,
   output [31:0] xor_res,
   output [31:0] not_res,
   output [31:0] logical_not_res,
   output [31:0] mux_res,
   output [31:0] rshift_res,
   output [31:0] lshift_res,
   output [31:0] andR_res,
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
   output [63:0] div_res,
   output [63:0] cat_res,
   output [1:0]  multiple_bitselect_res,
   output [95:0] fill_res
 );
   wire [38:0] _GEN_0 = {{7'd0}, x}; 
   wire [38:0] _lshift_res_T_1 = _GEN_0 << y[2:0]; 
   wire [63:0] _fill_res_T = {x,x}; 
   wire [31:0] _GEN_1 = x % y; 
   assign add_res = x + y; 
   assign sub_res = x - y; 
   assign per_res = _GEN_1[31:0]; 
   assign mul_res = x / y;
   assign and_res = x & y;
   assign or_res = x | y; 
   assign xor_res = x ^ y;
   assign not_res = ~x; 
   assign logical_not_res = {{31'd0}, x == 32'h0}; 
 endmodule
```

</td>
<td>

```scala mdoc:silent
class OperatorExampleModule extends Module {

  val x, y, c = IO(Input(UInt(32.W)))

  val add_res, sub_res, mod_res, div_res, and_res, or_res, xor_res, not_res,logical_not_res, mux_res,  rshift_res , lshift_res,  andR_res = IO(Output(UInt(32.W)))
  val logical_and_res, logical_or_res, equ_res, not_equ_res, orR_res, xorR_res, gt_res,lt_res, geq_res, leq_res,single_bitselect_res = IO(Output(Bool()))
  val mul_res, cat_res= IO(Output(UInt(64.W)))
  val multiple_bitselect_res = IO(Output(UInt(2.W)))
  val fill_res = IO(Output(UInt((3*32).W)))
  
  add_res := x + y
  sub_res := x - y 
  mod_res := x % y
  div_res:= x * y
  mul_res := x / y
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
ChiselStage.emitVerilog(new OperatorExampleModule)
```
</td>
         </tr>
    </table>
</body>
</html>
