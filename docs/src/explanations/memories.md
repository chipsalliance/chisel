---
layout: docs
title:  "Memories"
section: "chisel3"
---

# Memories

Chisel provides facilities for creating both read only and read/write memories.

## ROM

Users can define read-only memories by constructing a `Vec` with `VecInit`.
`VecInit` can accept either a variable-argument number of `Data` literals or a `Seq[Data]` literals that initialize the ROM.

For example, users can create a small ROM initialized to 1, 2, 4, 8 and loop through all values using a counter as an address generator as follows:

```scala mdoc:compile-only
import chisel3._
import chisel3.util.Counter
val m = VecInit(1.U, 2.U, 4.U, 8.U)
val c = Counter(m.length)
c.inc()
val r = m(c.value)
```

We can create an *n* value sine lookup table generator using a ROM initialized as follows:

```scala mdoc:compile-only
import chisel3._

val Pi = math.Pi
def sinTable(amp: Double, n: Int) = {
  val times =
    (0 until n).map(i => (i*2*Pi)/(n.toDouble-1) - Pi)
  val inits =
    times.map(t => Math.round(amp * math.sin(t)).asSInt(32.W))
  VecInit(inits)
}
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

![read/write ports example waveform](https://svg.wavedrom.com/github/freechipsproject/www.chisel-lang.org/master/docs/src/main/resources/json/smem_read_write.json)    


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

![read/write ports example waveform](https://svg.wavedrom.com/github/freechipsproject/www.chisel-lang.org/master/docs/src/main/resources/json/smem_rw.json)

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

Chisel supports multiple methods for annotating memories to be loaded from a text file containing hex or binary data. When using verilog simulation it uses the `$readmemh` or `$readmemb` verilog extension. The treadle simulator can also load memories using the same annotation.

#### Inline initialization with external file

Memories can be initialized by generating inline `readmemh` or `readmemb` statements in the output Verilog.

The function `loadMemoryFromFile` from `chisel3.util` allows the memory to be initialized by the synthesis software from the specified file. Chisel does not validate the file contents nor its location. Both the memory initialization file and the Verilog source should be accessible for the toolchain.

```scala mdoc:silent
import chisel3._
import chisel3.util.loadMemoryFromFile

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
    loadMemoryFromFile(mem, memoryFile)
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

By default, Chisel/Firrtl assumes an ASIC workflow and the inline initialization will generate the memory `readmem` statements inside an `ifndef SYNTHESIS` block which some synthesis tools (like Synplify and Yosys) define so the `readmem` statement is not read when inside this block.

To use a workflow suited to FPGAs, pass the command line argument `--target:fpga` to Chisel so the correct annotation is passed on to Verilog generation and the `readmem` statements are correctly placed and picked by FPGA synthesis tools.


There is also an experimental method to initialize memories by using an external SystemVerilog bind module. For more information, check the experimental docs on [Loading Memories](../appendix/experimental-features#loading-memories) feature.

