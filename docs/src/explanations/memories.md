---
layout: docs
title:  "Memories"
section: "chisel3"
---

# Memories

Chisel provides facilities for creating both read only and read/write memories.

## ROM

Users can define read only memories with a `Vec`:

```scala mdoc:invisible
import chisel3._
```

``` scala mdoc:compile-only
    VecInit(inits: Seq[T])
    VecInit(elt0: T, elts: T*)
```

where `inits` is a sequence of initial `Data` literals that initialize the ROM. For example,  users cancreate a small ROM initialized to 1, 2, 4, 8 and loop through all values using a counter as an address generator as follows:

``` scala mdoc:compile-only
    val m = VecInit(Array(1.U, 2.U, 4.U, 8.U))
    val r = m(counter(m.length.U))
```

We can create an *n* value sine lookup table using a ROM initialized as follows:

``` scala mdoc:silent
    def sinTable(amp: Double, n: Int) = {
      val times =
        (0 until n).map(i => (i*2*Pi)/(n.toDouble-1) - Pi)
      val inits =
        times.map(t => round(amp * sin(t)).asSInt(32.W))
      VecInit(inits)
    }
    def sinWave(amp: Double, n: Int) =
      sinTable(amp, n)(counter(n.U))
```

where `amp` is used to scale the fixpoint values stored in the ROM.

## Read-Write Memories

Memories are given special treatment in Chisel since hardware implementations of memory vary greatly. For example, FPGA memories are instantiated quite differently from ASIC memories. Chisel defines a memory abstraction that can map to either simple Verilog behavioural descriptions or to instances of memory modules that are available from external memory generators provided by foundry or IP vendors.


### `SyncReadMem`: sequential/synchronous-read, sequential/synchronous-write

Chisel has a construct called `SyncReadMem` for sequential/synchronous-read, sequential/synchronous-write memories. These `SyncReadMem`s will likely be synthesized to technology SRAMs (as opposed to register banks).

If the same memory address is both written and sequentially read on the same clock edge, or if a sequential read enable is cleared, then the read data is undefined.

Values on the read data port are not guaranteed to be held until the next read cycle. If that is the desired behavior, external logic to hold the last read value must be added.

#### Read port/write port
Ports into `SyncReadMem`s are created by applying a `UInt` index.  A 1024-entry SRAM with one write port and one read port might be expressed as follows:

```scala mdoc:silent
import chisel3._
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
```

Below is an example waveform of the one write port/one read port `SyncReadMem` with [masks](#masks). Note that the signal names will differ from the exact wire names generated for the `SyncReadMem`. With masking, it is also possible that multiple RTL arrays will be generated with the behavior below.

![read/write ports example waveform](https://svg.wavedrom.com/github/freechipsproject/www.chisel-lang.org/master/docs/src/main/tut/chisel3/memories_waveforms/smem_read_write.json)    

#### Single-ported
Single-ported SRAMs can be inferred when the read and write conditions are mutually exclusive in the same `when` chain:

```scala mdoc:silent
import chisel3._
class RWSmem extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })

  val mem = SyncReadMem(1024, UInt(width.W))
  io.dataOut := DontCare
  when(io.enable) {
    val rdwrPort = mem(io.addr)
    when (io.write) { rdwrPort := io.dataIn }
      .otherwise    { io.dataOut := rdwrPort }
  }
}
```

(The `DontCare` is there to make Chisel's [unconnected wire detection](unconnected-wires) aware that reading while writing is undefined.)

Here is an example single read/write port waveform, with [masks](#masks) (again, generated signal names and number of arrays may differ):

![read/write ports example waveform](https://svg.wavedrom.com/github/freechipsproject/www.chisel-lang.org/master/docs/src/main/tut/chisel3/memories_waveforms/smem_rw.json)

### `Mem`: combinational/asynchronous-read, sequential/synchronous-write

Chisel supports random-access memories via the `Mem` construct. Writes to `Mem`s are combinational/asynchronous-read, sequential/synchronous-write. These `Mem`s will likely be synthesized to register banks, since most SRAMs in modern technologies (FPGA, ASIC) tend to no longer support combinational (asynchronous) reads.

Creating asynchronous-read versions of the examples above simply involves replacing `SyncReadMem` with `Mem`.

### Masks

Chisel memories also support write masks for subword writes. Chisel will infer masks if the data type of the memory is a vector. To infer a mask, specify the `mask` argument of the `write` function which creates write ports. A given masked length is written if the corresponding mask bit is set. For example, in the example below, if the 0th bit of mask is true, it will write the lower byte of the data at corresponding address.

```scala mdoc:silent
import chisel3._
class MaskedReadWriteSmem extends Module {
  val width: Int = 8
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val mask = Input(Vec(4, Bool()))
    val dataIn = Input(Vec(4, UInt(width.W)))
    val dataOut = Output(Vec(4, UInt(width.W)))
  })

  // Create a 32-bit wide memory that is byte-masked
  val mem = SyncReadMem(1024, Vec(4, UInt(width.W)))
  // Write with mask
  mem.write(io.addr, io.dataIn, io.mask)
  io.dataOut := mem.read(io.addr, io.enable)
}
```

Here is an example of masks with readwrite ports:

```scala mdoc:silent
import chisel3._
class MaskedRWSmem extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val mask = Input(Vec(2, Bool()))
    val addr = Input(UInt(10.W))
    val dataIn = Input(Vec(2, UInt(width.W)))
    val dataOut = Output(Vec(2, UInt(width.W)))
  })

  val mem = SyncReadMem(1024, Vec(2, UInt(width.W)))
  io.dataOut := DontCare
  when(io.enable) {
    val rdwrPort = mem(io.addr)
    when (io.write) {
      when(io.mask(0)) {
        rdwrPort(0) := io.dataIn(0)
      }
      when(io.mask(1)) {
        rdwrPort(1) := io.dataIn(1)
      }
    }.otherwise { io.dataOut := rdwrPort }
  }
}
```

### Memory Initialization

Chisel memories can be initialized from an external binary or hex file emitting proper Verilog. There are multiple modes of initialization.

#### Inline initialization with external file

Memories can be initialized by generating inline `readmemh` or `readmemb` statements in the output Verilog.

The function `loadMemoryFromFileInline` from `chisel3.util.experimental` allows the memory to be initialized by the synthesis software from the specified file. Chisel does not validate the file contents nor it's location. Both the memory initialization file and the Verilog source should be accessible for the toolchain.

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

The inline initialization can generate the memory `readmem` statement inside a `ifndef SYNTHESIS` block, which suits ASIC workflow or outside the block suiting FPGA workflows.

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

Chisel also can initialize memories by generating a SV bind module with `readmemh` or `readmemb` statements by using the function `loadMemoryFromFile` from `chisel3.util.experimental`.

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

#### Inline initialization with embedded file contents

Memories can also be initialized embedding the file contents in the generated Verilog.

Data can be read from a single `BigInt` or an array of `BigInt` by using the following annotations:

```scala
// Read data into a BigInt
val src = Source.fromFile("filename")
val lines = src.getLines().toList
src.close()
val data = lines.map(_.trim).filter(_.nonEmpty).map(BigInt(_, radix))
val memory = Mem(32, UInt(32.W))

// Uses the above `data` BigInt
annotate(new ChiselAnnotation {
  override def toFirrtl = MemoryScalarInitAnnotation(memory.toTarget, data)
})

// Or read from an array
annotate(new ChiselAnnotation {
  override def toFirrtl = MemoryArrayInitAnnotation(memory.toTarget, Seq[0])
})
```

