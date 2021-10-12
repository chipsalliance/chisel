---
layout: docs
title:  "Experimental Features"
section: "chisel3"
---
# Experimental Features

Chisel has a number of new features that are worth checking out.  This page is an informal list of these features and projects.

- [FixedPoint](#fixed-point)
- [Module Variants](#module-variants)
- [Module Variants](#bundle-literals)
- [Interval Type](#interval-type)
- [Loading Memories for simulation or FPGA initialization](#loading-memories)


### FixedPoint  <a name="fixed-point"></a>
FixedPoint numbers are basic *Data* type along side of UInt, SInt, etc.  Most common math and logic operations
are supported. Chisel allows both the width and binary point to be inferred by the Firrtl compiler which can simplify
circuit descriptions. See [FixedPointSpec](https://github.com/freechipsproject/chisel3/tree/master/src/test/scala/chiselTests/FixedPointSpec.scala)

### Module Variants <a name="module-variants"></a>
The standard Chisel *Module* requires a `val io = IO(...)`, the experimental package introduces several
new ways of defining Modules
- BaseModule: no contents, instantiable
- BlackBox extends BaseModule
- UserDefinedModule extends BaseModule: this module can contain Chisel RTL. No default clock or reset lines. No default IO. - User should be able to specify non-io ports, ideally multiple of them.
- ImplicitModule extends UserModule: has clock, reset, and io, essentially current Chisel Module.
- RawModule: will be the user-facing version of UserDefinedModule
- Module: type-aliases to ImplicitModule, the user-facing version of ImplicitModule.

### Bundle Literals <a name="bundle-literals"></a>

Bundle literals can be constructed via an experimental import:

```scala mdoc
import chisel3._
import chisel3.experimental.BundleLiterals._

class MyBundle extends Bundle {
  val a = UInt(8.W)
  val b = Bool()
}

class Example extends RawModule {
  val out = IO(Output(new MyBundle))
  out := (new MyBundle).Lit(_.a -> 8.U, _.b -> true.B)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example)
```

Partial specification is allowed, defaulting any unconnected fields to 0 (regardless of type).

```scala mdoc
class Example2 extends RawModule {
  val out = IO(Output(new MyBundle))
  out := (new MyBundle).Lit(_.b -> true.B)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example2)
```

Bundle literals can also be nested arbitrarily.

```scala mdoc
class ChildBundle extends Bundle {
  val foo = UInt(8.W)
}

class ParentBundle extends Bundle {
  val a = UInt(8.W)
  val b = new ChildBundle
}

class Example3 extends RawModule {
  val out = IO(Output(new ParentBundle))
  out := (new ParentBundle).Lit(_.a -> 123.U, _.b -> (new ChildBundle).Lit(_.foo -> 42.U))
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example3)
```

### Vec Literals

Vec literals are very similar to Bundle literals and can be constructed via an experimental import.
They can be constructed in two forms, with type and length inferred as in: 

```scala mdoc
import chisel3._
import chisel3.experimental.VecLiterals._

class VecExample1 extends Module {
  val out = IO(Output(Vec(2, UInt(4.W))))
  out := Vec.Lit(0xa.U, 0xbb.U)
}
```
```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new VecExample1)
```

or explicitly as in:

```scala mdoc
import chisel3._
import chisel3.experimental.VecLiterals._

class VecExample1a extends Module {
  val out = IO(Output(Vec(2, UInt(4.W))))
  out := Vec(2, UInt(4.W)).Lit(0 -> 1.U, 1 -> 2.U)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new VecExample1a)
```

The following examples all use the explicit form.
With the explicit form partial specification is allowed.
When used with as a `Reg` `reset` value, only specified indices of the `Reg`'s `Vec`
will be reset

```scala mdoc
class VecExample2 extends RawModule {
  val out = IO(Output(Vec(4, UInt(4.W))))
  out := Vec(4, UInt(4.W)).Lit(0 -> 1.U, 3 -> 7.U)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new VecExample2)
```

Registers can be initialized from Vec literals

```scala mdoc
class VecExample3 extends Module {
  val out = IO(Output(Vec(4, UInt(8.W))))
  val y = RegInit(
    Vec(4, UInt(8.W)).Lit(0 -> 0xAB.U(8.W), 1 -> 0xCD.U(8.W), 2 -> 0xEF.U(8.W), 3 -> 0xFF.U(8.W))
  )
  out := y
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new VecExample3)
```

Vec literals can also be nested arbitrarily.

```scala mdoc
class VecExample5 extends RawModule {
  val out = IO(Output(Vec(2, new ChildBundle)))
  out := Vec(2, new ChildBundle).Lit(
    0 -> (new ChildBundle).Lit(_.foo -> 42.U),
    1 -> (new ChildBundle).Lit(_.foo -> 7.U)
  )
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new VecExample5)
```

### Interval Type <a name="interval-type"></a>

**Intervals** are a new experimental numeric type that comprises UInt, SInt and FixedPoint numbers.
It augments these types with range information, i.e. upper and lower numeric bounds.
This information can be used to exercise tighter programmatic control over the ultimate widths of
signals in the final circuit.  The **Firrtl** compiler can infer this range information based on
operations and earlier values in the circuit. Intervals support all the ordinary bit and arithmetic operations
associated with UInt, SInt, and FixedPoint and adds the following methods for manipulating the range of
a **source** Interval with the IntervalRange of **target** Interval

#### Clip -- Fit the value **source** into the IntervalRange of **target**, saturate if out of bounds
The clip method applied to an interval creates a new interval based on the argument to clip,
and constructs the necessary hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be pegged to either maximum or minimum of result range as appropriate.

> Generates necessary hardware to clip values, values greater than range are set to range.high, values lower than range are set to range min.

#### Wrap -- Fit the value **source** into the IntervalRange of **target**, wrapping around if out of bounds
The wrap method applied to an interval creates a new interval based on the argument to wrap,
and constructs the necessary
hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be wrapped until they fall within the result range.

> Generates necessary hardware to wrap values, values greater than range are set to range.high, values lower than range are set to range min.

> Does not handle out of range values that are less than half the minimum or greater than twice maximum

#### Squeeze -- Fit the value **source** into the smallest IntervalRange based on source and target.
The squeeze method applied to an interval creates a new interval based on the argument to clip, the two ranges must overlap
behavior of squeeze with inputs outside of the produced range is undefined.

> Generates no hardware, strictly a sizing operation

##### Range combinations

| Condition | A.clip(B) | A.wrap(B) | A.squeeze(B) |
| --------- | --------------- | --------------- | --------------- |
| A === B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A contains B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| B contains A   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min < B min, A max in B  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min in B, A max > B max  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A strictly less than B   | error               | error               | error               |
| A strictly greater than B   | error               | error               | error               |


#### Applying binary point operators to an Interval

Consider a Interval with a binary point of 3: aaa.bbb

| operation | after operation | binary point | lower | upper | meaning |
| --------- | --------------- | ------------ | ----- | ----- | ------- |
| setBinaryPoint(2) | aaa.bb |  2 | X | X  | set the precision |
| shiftLeftBinaryPoint(2) | a.aabbb |  5 | X | X  | increase the precision |
| shiftRighBinaryPoint(2) | aaaa.b |  1 | X | X  | reduce the precision |

## Loading Memories for simulation or FPGA initialization <a name="loading-memories"></a>

Chisel supports multiple experimental methods for annotating memories to be loaded from a text file containing hex or binary data. When using verilog simulation it uses the `$readmemh` or `$readmemb` verilog extension. The treadle simulator can also load memories using the same annotation.

### Inline initialization with external file

Memories can be initialized by generating inline `readmemh` or `readmemb` statements in the output Verilog.

The function `loadMemoryFromFileInline` from `chisel3.util.experimental` allows the memory to be initialized by the synthesis software from the specified file. Chisel does not validate the file contents nor its location. Both the memory initialization file and the Verilog source should be accessible for the toolchain.

```scala mdoc:silent
import chisel3._
import chisel3.util.experimental.loadMemoryFromFileInline

class InitMemInline(memoryFile: String = "") extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })

  val mem = SyncReadMem(1024, UInt(width.W))
  // Initialize memory
  if (memoryFile.trim().nonEmpty) {
    loadMemoryFromFileInline(mem, memoryFile)
  }
  io.dataOut := DontCare
  when(io.enable) {
    val rdwrPort = mem(io.addr)
    when (io.write) { rdwrPort := io.dataIn }
      .otherwise    { io.dataOut := rdwrPort }
  }
}
```

The default is to use `$readmemh` (which assumes all numbers in the file are in ascii hex),
but to use ascii binary there is an optional `hexOrBinary` argument which can be set to `MemoryLoadFileType.Hex` or `MemoryLoadFileType.Binary`. You will need to add an additional import.

By default, the inline initialization will generate the memory `readmem` statements inside an `ifndef SYNTHESIS` block, which suits ASIC workflow.

Some synthesis tools (like Synplify and Yosys) define `SYNTHESIS` so the `readmem` statement is not read when inside this block.

To control this, one can use the `MemoryNoSynthInit` and `MemorySynthInit` annotations from `firrtl.annotations`. The former which is the default setting when no annotation is present generates `readmem` inside the block. Using the latter, the statement are generated outside the `ifndef` block so it can be used by FPGA synthesis tools.

Below an example for initialization suited for FPGA workflows:

```scala mdoc:silent
import chisel3._
import chisel3.util.experimental.loadMemoryFromFileInline
import chisel3.experimental.{annotate, ChiselAnnotation}
import firrtl.annotations.MemorySynthInit

class InitMemInlineFPGA(memoryFile: String = "") extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })

  // Notice the annotation below
  annotate(new ChiselAnnotation {
    override def toFirrtl =
      MemorySynthInit
  })

  val mem = SyncReadMem(1024, UInt(width.W))
  if (memoryFile.trim().nonEmpty) {
    loadMemoryFromFileInline(mem, memoryFile)
  }
  io.dataOut := DontCare
  when(io.enable) {
    val rdwrPort = mem(io.addr)
    when (io.write) { rdwrPort := io.dataIn }
      .otherwise    { io.dataOut := rdwrPort }
  }
}
```

#### SystemVerilog Bind Initialization

Chisel can also initialize memories by generating a SV bind module with `readmemh` or `readmemb` statements by using the function `loadMemoryFromFile` from `chisel3.util.experimental`.

```scala mdoc:silent
import chisel3._
import chisel3.util.experimental.loadMemoryFromFile

class InitMemBind(val bits: Int, val size: Int, filename: String) extends Module {
  val io = IO(new Bundle {
    val nia = Input(UInt(bits.W))
    val insn = Output(UInt(32.W))
  })

  val memory = Mem(size, UInt(32.W))
  io.insn := memory(io.nia >> 2);
  loadMemoryFromFile(memory, filename)
}
```

Which generates the bind module:

```verilog
module BindsTo_0_Foo(
  input         clock,
  input         reset,
  input  [31:0] io_nia,
  output [31:0] io_insn
);

initial begin
  $readmemh("test.hex", Foo.memory);
end
endmodule

bind Foo BindsTo_0_Foo BindsTo_0_Foo_Inst(.*);
```

### Notes on files

There is no simple answer to where to put the `hex` or `bin` file with the initial contents. It's probably best to create a resource directory somewhere and reference that through a full path or place the file beside the generated Verilog. Another option is adding the path to the memory file in the synthesis tool path. Because these files may be large, Chisel does not copy them.
> Don't forget there is no decimal option, so a 10 in an input file will be 16 decimal

See: [ComplexMemoryLoadingSpec.scala](https://github.com/freechipsproject/chisel-testers/blob/master/src/test/scala/examples/ComplexMemoryLoadingSpec.scala) and
[LoadMemoryFromFileSpec.scala](https://github.com/freechipsproject/chisel-testers/blob/master/src/test/scala/examples/LoadMemoryFromFileSpec.scala)
for working examples.


### Aggregate memories

Aggregate memories are supported but in bit of a clunky way. Since they will be split up into a memory per field, the following convention was adopted.  When specifying the file for such a memory the file name should be regarded as a template. If the memory is a Bundle e.g.

```scala mdoc:compile-only
class MemDataType extends Bundle {
  val a = UInt(16.W)
  val b = UInt(32.W)
  val c = Bool()
}
```

The memory will be split into `memory_a`, `memory_b`, and `memory_c`. Similarly if a load file is specified as `"memory-load.txt"` the simulation will expect that there will be three files, `"memory-load_a.txt"`, `"memory-load_b.txt"`, `"memory-load_c.txt"`

> Note: The use of `_` and that the memory field name is added before any file suffix. The suffix is optional but if present is considered to be the text after the last `.` in the file name.
