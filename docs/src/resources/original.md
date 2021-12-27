<!Doctype html>
<html>
<title> Two side column </title>
<body>
    <script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>
    <table border ="0">
        <tr>
             <td><b style="font-size:30px">Title</b></td>
            <td><b style="font-size:30px">Title 2</b></td>
         </tr>
         <tr>
            <td> hello</td>
            <td>
# Deep Dive into Connection Operators

Chisel contains two connection operators, `:=` and `<>`. This document provides a deeper explanation of the differences of the two and when to use one or the other. The differences are demonstrated with experiments using Scastie examples which use `DecoupledIO`.



### Experiment Setup

```scala
// Imports used by the following examples
import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.DecoupledIO
```

The diagram for the experiment can be viewed [here](https://docs.google.com/document/d/14C918Hdahk2xOGSJJBT-ZVqAx99_hg3JQIq-vaaifQU/edit?usp=sharing).
![Experiment Image](https://raw.githubusercontent.com/chipsalliance/chisel3/master/docs/src/images/chisel_01.png?sanitize=true)

```scala

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  // connect Producer to IO
  p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  // connect consumer to IO
  io.out <> c.io.b
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.b <> io.a
}
```
Below we can see the resulting Verilog for this example:
```scala
ChiselStage.emitVerilog(new Wrapper)
// res0: String = """module PipelineStage(
//   output       io_a_ready,
//   input        io_a_valid,
//   input  [7:0] io_a_bits,
//   input        io_b_ready,
//   output       io_b_valid,
//   output [7:0] io_b_bits
// );
//   assign io_a_ready = io_b_ready; // @[two-side-column.md 41:8]
//   assign io_b_valid = io_a_valid; // @[two-side-column.md 41:8]
//   assign io_b_bits = io_a_bits; // @[two-side-column.md 41:8]
// endmodule
// module Wrapper(
//   input        clock,
//   input        reset,
//   output       io_in_ready,
//   input        io_in_valid,
//   input  [7:0] io_in_bits,
//   input        io_out_ready,
//   output       io_out_valid,
//   output [7:0] io_out_bits
// );
//   wire  p_io_a_ready; // @[two-side-column.md 25:17]
//   wire  p_io_a_valid; // @[two-side-column.md 25:17]
//   wire [7:0] p_io_a_bits; // @[two-side-column.md 25:17]
//   wire  p_io_b_ready; // @[two-side-column.md 25:17]
//   wire  p_io_b_valid; // @[two-side-column.md 25:17]
//   wire [7:0] p_io_b_bits; // @[two-side-column.md 25:17]
//   wire  c_io_a_ready; // @[two-side-column.md 26:17]
//   wire  c_io_a_valid; // @[two-side-column.md 26:17]
//   wire [7:0] c_io_a_bits; // @[two-side-column.md 26:17]
//   wire  c_io_b_ready; // @[two-side-column.md 26:17]
//   wire  c_io_b_valid; // @[two-side-column.md 26:17]
//   wire [7:0] c_io_b_bits; // @[two-side-column.md 26:17]
//   PipelineStage p ( // @[two-side-column.md 25:17]
//     .io_a_ready(p_io_a_ready),
//     .io_a_valid(p_io_a_valid),
//     .io_a_bits(p_io_a_bits),
//     .io_b_ready(p_io_b_ready),
//     .io_b_valid(p_io_b_valid),
//     .io_b_bits(p_io_b_bits)
//   );
//   PipelineStage c ( // @[two-side-column.md 26:17]
//     .io_a_ready(c_io_a_ready),
//     .io_a_valid(c_io_a_valid),
//     .io_a_bits(c_io_a_bits),
//     .io_b_ready(c_io_b_ready),
//     .io_b_valid(c_io_b_valid),
//     .io_b_bits(c_io_b_bits)
// ...
```
</td>
         </tr>
    </table>
</body>


</html>
