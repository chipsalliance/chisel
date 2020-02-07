---
layout: docs
title:  "Memories"
section: "chisel3"
---
Chisel provides facilities for creating both read only and read/write memories.

## ROM

Users can define read only memories with a `Vec`:

``` scala
    VecInit(inits: Seq[T])
    VecInit(elt0: T, elts: T*)
```

where `inits` is a sequence of initial `Data` literals that initialize the ROM. For example,  users cancreate a small ROM initialized to 1, 2, 4, 8 and loop through all values using a counter as an address generator as follows:

``` scala
    val m = VecInit(Array(1.U, 2.U, 4.U, 8.U))
    val r = m(counter(m.length.U))
```

We can create an *n* value sine lookup table using a ROM initialized as follows:

``` scala
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

## Mem

Memories are given special treatment in Chisel since hardware implementations of memory vary greatly. For example, FPGA memories are instantiated quite differently from ASIC memories. Chisel defines a memory abstraction that can map to either simple Verilog behavioural descriptions or to instances of memory modules that are available from external memory generators provided by foundry or IP vendors.

Chisel supports random-access memories via the `Mem` construct. Writes to `Mem`s are **combinational/asynchronous-read, sequential/synchronous-write**. These `Mem`s will likely be synthesized to register banks, since most SRAMs in modern technologies (FPGA, ASIC) tend to no longer support combinational (asynchronous) reads.

Chisel also has a construct called `SyncReadMem` for **sequential/synchronous-read, sequential/synchronous-write** memories. These `SyncReadMem`s will likely be synthesized to technology SRAMs (as opposed to register banks).

Ports into Mems are created by applying a `UInt` index.  A 1024-entry register file with one write port and one sequential/synchronous read port might be expressed as follows:

```scala
val width:Int = 32
val addr = Wire(UInt(width.W))
val dataIn = Wire(UInt(width.W))
val dataOut = Wire(UInt(width.W))
val enable = Wire(Bool())

// assign data...

// Create a synchronous-read, synchronous-write memory (like in FPGAs).
val mem = SyncReadMem(1024, UInt(width.W))
// Create one write port and one read port.
mem.write(addr, dataIn)
dataOut := mem.read(addr, enable)
```
Creating an asynchronous-read version of the above simply involves replacing `SyncReadMem` with just `Mem`.

Chisel can also infer other features such as single ports and masks directly with Mem.

Single-ported SRAMs can be inferred when the read and write conditions are
mutually exclusive in the same `when` chain:

```scala mdoc:silent
import chisel3._
class ReadWriteSMEM extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val addr = Input(UInt(10.W))
    val dataIn = Input(UInt(width.W))
    val dataOut = Output(UInt(width.W))
  })


  val mem = SyncReadMem(2048, UInt(32.W))

  io.dataOut := DontCare
  when(io.enable) {
    val rdwrPort = mem(io.addr)
    when (io.write) { rdwrPort := io.dataIn }
      .otherwise    { io.dataOut := rdwrPort }
  }

}
```

(The `DontCare` is there to make Chisel's [unconnected wire detection](unconnected-wires) aware that reading while writing is undefined.)

If the same `Mem` address is both written and sequentially read on the same clock
edge, or if a sequential read enable is cleared, then the read data is
undefined.

### Masks

Chisel memories also support write masks for subword writes. Chisel will infer masks if the data type of the memory is a vector. To infer a mask, specify the `mask` argument of the `write` function which creates write ports. A given masked length is written if the corresponding mask bit is set. For example, in the example below, if the 0th bit of mask is true, it will write the lower 8 bits of the corresponding address.

```scala
val dataOut = Wire(Vec(4, UInt(8.W)))
val dataIn = Wire(Vec(4, UInt(8.W)))
val mask = Wire(Vec(4, Bool()))
val enable = Wire(Bool())
val readAddr = Wire(UInt(10.W))
val writeAddr = Wire(UInt(10.W))

// ... assign values ...

// Create a 32-bit wide memory that is byte-masked.
val mem = SyncReadMem(1024, Vec(4, UInt(8.W)))
// Create one masked write port and one read port.
mem.write(writeAddr, dataIn, mask)
dataOut := mem.read(readAddr, enable)
```

Here is an example of masks with readwrite ports:

```scala mdoc:silent
import chisel3._
// Chisel Code: Declare a new module definition
class ReadWriteSMEMWithMask extends Module {
  val width: Int = 32
  val io = IO(new Bundle {
    val enable = Input(Bool())
    val write = Input(Bool())
    val mask = Input(Vec(2, Bool()))
    val addr = Input(UInt(10.W))
    val dataIn = Input(Vec(2, UInt(width.W)))
    val dataOut = Output(Vec(2, UInt(width.W)))
  })


  val mem = SyncReadMem(2048, Vec(2, UInt(32.W)))

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
