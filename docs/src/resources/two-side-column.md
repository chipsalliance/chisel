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

#Parameterizing a Module

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

#Wire assignment

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

#Register assignment

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

#Case statement

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
class DelayBy1(resetValue: Option[UInt] = None) extends Module {
  val io = IO(new Bundle {
    val in  = Input( UInt(16.W))
    val out = Output(UInt(16.W))
  })
  val reg = resetValue match {
    case Some(r) => RegInit(r)
    case None    => Reg(UInt())
  }
  reg := io.in
  io.out := reg
}

println(getVerilogString(new DelayBy1))
println(getVerilogString(new DelayBy1(Some(3.U))))
```
</td>
<td>

```
module DelayBy1(
  input         clock,
  input         reset,
  input  [15:0] io_in,
  output [15:0] io_out
);
`ifdef RANDOMIZE_REG_INIT
  reg [31:0] _RAND_0;
`endif // RANDOMIZE_REG_INIT
  reg [15:0] reg_; // @[main.scala 15:24]
  assign io_out = reg_; // @[main.scala 18:10]
  always @(posedge clock) begin
    reg_ <= io_in; // @[main.scala 17:7]
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
  reg_ = _RAND_0[15:0];
`endif // RANDOMIZE_REG_INIT
  `endif // RANDOMIZE
end // initial
`ifdef FIRRTL_AFTER_INITIAL
`FIRRTL_AFTER_INITIAL
`endif
`endif // SYNTHESIS
endmodule

module DelayBy1(
input         clock,
input         reset,
input  [15:0] io_in,
output [15:0] io_out
);
`ifdef RANDOMIZE_REG_INIT
reg [31:0] _RAND_0;
`endif // RANDOMIZE_REG_INIT
reg [15:0] reg_; // @[main.scala 14:28]
assign io_out = reg_; // @[main.scala 18:10]
always @(posedge clock) begin
if (reset) begin // @[main.scala 14:28]
reg_ <= 16'h3; // @[main.scala 14:28]
end else begin
reg_ <= io_in; // @[main.scala 17:7]
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
reg_ = _RAND_0[15:0];
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

#systemVerilog Interfaces

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