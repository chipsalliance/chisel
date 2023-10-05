---
layout: docs
title:  "Blackboxes"
section: "chisel3"
---

# BlackBoxes

Chisel *BlackBoxes* are used to instantiate externally defined modules. This construct is useful
for hardware constructs that cannot be described in Chisel and for connecting to FPGA or other IP not defined in Chisel.

Modules defined as a `BlackBox` will be instantiated in the generated Verilog, but no code
will be generated to define the behavior of module.

Unlike Module, `BlackBox` has no implicit clock and reset.
`BlackBox`'s clock and reset ports must be explicitly declared and connected to input signals.
Ports declared in the IO Bundle will be generated with the requested name (ie. no preceding `io_`).

### Parameterization

Verilog parameters can be passed as an argument to the BlackBox constructor.

For example, consider instantiating a Xilinx differential clock buffer (IBUFDS) in a Chisel design:

```scala mdoc:silent
import chisel3._
import chisel3.util._
import chisel3.experimental._ // To enable experimental features

class IBUFDS extends BlackBox(Map("DIFF_TERM" -> "TRUE",
                                  "IOSTANDARD" -> "DEFAULT")) {
  val io = IO(new Bundle {
    val O = Output(Clock())
    val I = Input(Clock())
    val IB = Input(Clock())
  })
}

class Top extends Module {
  val io = IO(new Bundle {})
  val ibufds = Module(new IBUFDS)
  // connecting one of IBUFDS's input clock ports to Top's clock signal
  ibufds.io.I := clock
}
```

In the Chisel-generated Verilog code, `IBUFDS` will be instantiated as:

```verilog
IBUFDS #(.DIFF_TERM("TRUE"), .IOSTANDARD("DEFAULT")) ibufds (
  .IB(ibufds_IB),
  .I(ibufds_I),
  .O(ibufds_O)
);
```

### Providing Implementations for Blackboxes

Chisel provides the following ways of delivering the code underlying the blackbox.  Consider the following blackbox that
 adds two real numbers together.  The numbers are represented in chisel3 as 64-bit unsigned integers.

```scala mdoc:silent:reset
import chisel3._
class BlackBoxRealAdd extends BlackBox {
  val io = IO(new Bundle {
    val in1 = Input(UInt(64.W))
    val in2 = Input(UInt(64.W))
    val out = Output(UInt(64.W))
  })
}
```

The implementation is described by the following verilog

```verilog
module BlackBoxRealAdd(
    input  [63:0] in1,
    input  [63:0] in2,
    output reg [63:0] out
);
  always @* begin
    out <= $realtobits($bitstoreal(in1) + $bitstoreal(in2));
  end
endmodule
```

### Blackboxes with Verilog in a Resource File

In order to deliver the verilog snippet above to the backend simulator, chisel3 provides the following tools based on the chisel/firrtl [annotation system](../explanations/annotations).  Add the trait `HasBlackBoxResource` to the declaration, and then call a function in the body to say where the system can find the verilog.  The Module now looks like

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.HasBlackBoxResource

class BlackBoxRealAdd extends BlackBox with HasBlackBoxResource {
  val io = IO(new Bundle {
    val in1 = Input(UInt(64.W))
    val in2 = Input(UInt(64.W))
    val out = Output(UInt(64.W))
  })
  addResource("/real_math.v")
}
```

The verilog snippet above gets put into a resource file names `real_math.v`.  What is a resource file? It comes from
 a java convention of keeping files in a project that are automatically included in library distributions. In a typical
 chisel3 project, see [chisel-template](https://github.com/ucb-bar/chisel-template), this would be a directory in the
 source hierarchy: `src/main/resources/real_math.v`.

### Blackboxes with In-line Verilog
It is also possible to place this verilog directly in the scala source.  Instead of `HasBlackBoxResource` use
 `HasBlackBoxInline` and instead of `setResource` use `setInline`.  The code will look like this.

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.HasBlackBoxInline
class BlackBoxRealAdd extends BlackBox with HasBlackBoxInline {
  val io = IO(new Bundle {
    val in1 = Input(UInt(64.W))
    val in2 = Input(UInt(64.W))
    val out = Output(UInt(64.W))
  })
  setInline("BlackBoxRealAdd.v",
    """module BlackBoxRealAdd(
      |    input  [15:0] in1,
      |    input  [15:0] in2,
      |    output [15:0] out
      |);
      |always @* begin
      |  out <= $realtobits($bitstoreal(in1) + $bitstoreal(in2));
      |end
      |endmodule
    """.stripMargin)
}
```

This technique will copy the inline verilog into the target directory under the name `BlackBoxRealAdd.v`

### Under the Hood
This mechanism of delivering verilog content to the testing backends is implemented via chisel/firrtl annotations. The
two methods, inline and resource, are two kinds of annotations that are created via the `setInline` and
`setResource` methods calls.  Those annotations are passed through to the chisel-testers which in turn passes them
on to firrtl.  The default firrtl verilog compilers have a pass that detects the annotations and moves the files or
inline test into the build directory.  For each unique file added, the transform adds a line to a file
`black_box_verilog_files.f`, this file is added to the command line constructed for verilator or vcs to inform them where
to look.
The [dsptools project](https://github.com/ucb-bar/dsptools) is a good example of using this feature to build a real
number simulation tester based on black boxes.

