---
layout: docs
title: |
    External Modules ("Black Boxes")
section: "chisel3"
---

# External Modules ("Black Boxes")

Chisel *External Modules* are used to instantiate externally defined modules. This construct is useful
for hardware constructs that cannot be described in Chisel and for connecting to FPGA or other IP not defined in Chisel.

Modules defined as an `ExtModule` will be instantiated in the generated Verilog, but no code
will be generated to define the behavior of module.

Unlike Module, `ExtModule` has no implicit clock and reset.
Instead, they behave like `RawModule` in this regard.
`ExtModule`'s clock and reset ports must be explicitly declared and connected to input signals.

### Parameterization

Verilog parameters can be passed as an argument to the `ExtModule` constructor.

For example, consider instantiating a Xilinx differential clock buffer (IBUFDS) in a Chisel design:

```scala mdoc:silent
import chisel3._
import chisel3.util._
import chisel3.experimental.fromStringToStringParam

class IBUFDS extends ExtModule(Map("DIFF_TERM" -> "TRUE",
                                  "IOSTANDARD" -> "DEFAULT")) {
  val O = IO(Output(Clock()))
  val I = IO(Input(Clock()))
  val IB = IO(Input(Clock()))
}

class Top extends Module {
  val ibufds = Module(new IBUFDS)
  // connecting one of IBUFDS's input clock ports to Top's clock signal
  ibufds.I := clock
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

### Providing Implementations for External Modules

Chisel provides the following ways of delivering the code underlying the external module. Consider the following external module that
adds two real numbers together. The numbers are represented in chisel3 as 64-bit unsigned integers.

```scala mdoc:silent:reset
import chisel3._
class ExtModuleRealAdd extends ExtModule {
  val in1 = IO(Input(UInt(64.W)))
  val in2 = IO(Input(UInt(64.W)))
  val out = IO(Output(UInt(64.W)))
}
```

The implementation is described by the following verilog

```verilog
module ExtModuleRealAdd(
    input  [63:0] in1,
    input  [63:0] in2,
    output reg [63:0] out
);
  always @* begin
    out <= $realtobits($bitstoreal(in1) + $bitstoreal(in2));
  end
endmodule
```

### External Modules with Verilog in a Resource File

In order to deliver the Verilog snippet above to the backend simulator, chisel3 provides the following tools basedf on the Chisel/FIRRTL [annotation system](../explanations/annotations) via methods that are already available on `ExtModule`.  To include a Java resource, use `addResource`:

```scala mdoc:silent:reset
import chisel3._

class ExtModuleRealAdd extends ExtModule {
  val in1 = IO(Input(UInt(64.W)))
  val in2 = IO(Input(UInt(64.W)))
  val out = IO(Output(UInt(64.W)))
  addResource("/real_math.v")
}
```

The verilog snippet above gets put into a resource file names `real_math.v`.  What is a resource file? It comes from
 a java convention of keeping files in a project that are automatically included in library distributions. In a typical
 Chisel project, see [chisel-template](https://github.com/chipsalliance/chisel-template), this would be a directory in the
 source hierarchy: `src/main/resources/real_math.v`.

### External Modules with In-line Verilog
It is also possible to place this Verilog directly in the scala source.  Instead
of `addResource` use `setInline`.  The code will look like this:

```scala mdoc:silent:reset
import chisel3._
class ExtModuleRealAdd extends ExtModule {
  val in1 = IO(Input(UInt(64.W)))
  val in2 = IO(Input(UInt(64.W)))
  val out = IO(Output(UInt(64.W)))
  setInline("ExtModuleRealAdd.v",
    """module ExtModuleRealAdd(
      |    input  [63:0] in1,
      |    input  [63:0] in2,
      |    output reg [63:0] out
      |);
      |always @* begin
      |  out <= $realtobits($bitstoreal(in1) + $bitstoreal(in2));
      |end
      |endmodule
    """.stripMargin)
}
```

This technique will copy the inline verilog into the target directory under the name `ExtModuleRealAdd.v`

### Under the Hood
This mechanism of delivering verilog content to the testing backends is implemented via Chisel/FIRRTL annotations. The
two methods, inline and resource, are two kinds of annotations that are created via the `setInline` and
`addResource` methods calls.  Those annotations are passed through to the chisel-testers which in turn passes them
on to firrtl.  The default firrtl verilog compilers have a pass that detects the annotations and moves the files or
inline test into the build directory.  For each unique file added, the transform adds a line to a file
`black_box_verilog_files.f`, this file is added to the command line constructed for verilator or vcs to inform them where
to look.
The [dsptools project](https://github.com/ucb-bar/dsptools) is a good example of using this feature to build a real
number simulation tester based on black boxes.
