---
layout: docs
title:  "Polymorphism and Parameterization"
section: "chisel3"
---

# Polymorphism and Parameterization

_This section is advanced and can be skipped at first reading._

Scala is a strongly typed language and uses parameterized types to specify generic functions and classes.
In this section, we show how Chisel users can define their own reusable functions and classes using parameterized classes.

## Parameterized Functions

Earlier we defined `Mux2` on `Bool`, but now we show how we can define a generic multiplexer function.
We define this function as taking a boolean condition and con and alt arguments (corresponding to then and else expressions) of type `T`:

```scala
def Mux[T <: Bits](c: Bool, con: T, alt: T): T = { ... }
```

where `T` is required to be a subclass of `Bits`.
Scala ensures that in each usage of `Mux`, it can find a common superclass of the actual con and alt argument types,
otherwise it causes a Scala compilation type error.
For example,

```scala
Mux(c, UInt(10), UInt(11))
```

yields a `UInt` wire because the `con` and `alt` arguments are each of type `UInt`.

<!---
Jack: I cannot seem to get this to actually work
      Scala does not like the * in FIR since it could be from UInt or SInt

We now present a more advanced example of parameterized functions for defining an inner product FIR digital filter generically over Chisel `Num`s.

The inner product FIR filter can be mathematically defined as:
\begin{equation}
y[t] = \sum_j w_j * x_j[t-j]
\end{equation}


where `x` is the input and `w` is a vector of weights.
In Chisel this can be defined as:


```scala
def delays[T <: Data](x: T, n: Int): List[T] =
  if (n <= 1) List(x) else x :: delays(RegNext(x), n - 1)

def FIR[T <: Data with Num[T]](ws: Seq[T], x: T): T =
  ws zip delays(x, ws.length) map { case (a, b) => a * b } reduce (_ + _)
```

where
`delays` creates a list of incrementally increasing delays of its input and
`reduce` constructs a reduction circuit given a binary combiner function `f`.
In this case, `reduce` creates a summation circuit.
Finally, the `FIR` function is constrained to work on inputs of type `Num` where Chisel multiplication and addition are defined.
--->

## Parameterized Classes

Like parameterized functions, we can also parameterize classes to make them more reusable.
For instance, we can generalize the Filter class to use any kind of link.
We do so by parameterizing the `FilterIO` class and defining the constructor to take a single argument `gen` of type `T` as below.
```scala mdoc:invisible
import chisel3._
```
```scala mdoc:silent
class FilterIO[T <: Data](gen: T) extends Bundle {
  val x = Input(gen)
  val y = Output(gen)
}
```

We can now define `Filter` by defining a module class that also takes a link type constructor argument and passes it through to the `FilterIO` interface constructor:

```scala mdoc:silent
class Filter[T <: Data](gen: T) extends Module {
  val io = IO(new FilterIO(gen))
  // ...
}
```

We can now define a `PLink`-based `Filter` as follows:
```scala mdoc:invisible
class SimpleLink extends Bundle {
  val data = Output(UInt(16.W))
  val valid = Output(Bool())
}
class PLink extends SimpleLink {
  val parity = Output(UInt(5.W))
}
```
```scala mdoc:compile-only
val f = Module(new Filter(new PLink))
```

A generic FIFO could be defined as follows:

```scala mdoc:silent
import chisel3.util.log2Up

class DataBundle extends Bundle {
  val a = UInt(32.W)
  val b = UInt(32.W)
}

class Fifo[T <: Data](gen: T, n: Int) extends Module {
  val io = IO(new Bundle {
    val enqVal = Input(Bool())
    val enqRdy = Output(Bool())
    val deqVal = Output(Bool())
    val deqRdy = Input(Bool())
    val enqDat = Input(gen)
    val deqDat = Output(gen)
  })
  val enqPtr     = RegInit(0.U((log2Up(n)).W))
  val deqPtr     = RegInit(0.U((log2Up(n)).W))
  val isFull     = RegInit(false.B)
  val doEnq      = io.enqRdy && io.enqVal
  val doDeq      = io.deqRdy && io.deqVal
  val isEmpty    = !isFull && (enqPtr === deqPtr)
  val deqPtrInc  = deqPtr + 1.U
  val enqPtrInc  = enqPtr + 1.U
  val isFullNext = Mux(doEnq && ~doDeq && (enqPtrInc === deqPtr),
                         true.B, Mux(doDeq && isFull, false.B,
                         isFull))
  enqPtr := Mux(doEnq, enqPtrInc, enqPtr)
  deqPtr := Mux(doDeq, deqPtrInc, deqPtr)
  isFull := isFullNext
  val ram = Mem(n, gen)
  when (doEnq) {
    ram(enqPtr) := io.enqDat
  }
  io.enqRdy := !isFull
  io.deqVal := !isEmpty
  ram(deqPtr) <> io.deqDat
}
```

An Fifo with 8 elements of type DataBundle could then be instantiated as:

```scala mdoc:compile-only
val fifo = Module(new Fifo(new DataBundle, 8))
```

It is also possible to define a generic decoupled (ready/valid) interface:
```scala mdoc:invisible:reset
import chisel3._
class DataBundle extends Bundle {
  val a = UInt(32.W)
  val b = UInt(32.W)
}
```

```scala mdoc:silent
class DecoupledIO[T <: Data](data: T) extends Bundle {
  val ready = Input(Bool())
  val valid = Output(Bool())
  val bits  = Output(data)
}
```

This template can then be used to add a handshaking protocol to any
set of signals:

```scala mdoc:silent
class DecoupledDemo extends DecoupledIO(new DataBundle)
```

The FIFO interface can be now be simplified as follows:

```scala mdoc:silent
class Fifo[T <: Data](data: T, n: Int) extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(new DecoupledIO(data))
    val deq = new DecoupledIO(data)
  })
  // ...
}
```

## Parametrization based on Modules

You can also parametrize modules based on other modules rather than just types. The following is an example of a module parametrized by other modules as opposed to e.g. types.

```scala mdoc:silent
import chisel3.RawModule
import chisel3.experimental.BaseModule
import circt.stage.ChiselStage

// Provides a more specific interface since generic Module
// provides no compile-time information on generic module's IOs.
trait MyAdder {
    def in1: UInt
    def in2: UInt
    def out: UInt
}

class Mod1 extends RawModule with MyAdder {
    val in1 = IO(Input(UInt(8.W)))
    val in2 = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    out := in1 + in2
}

class Mod2 extends RawModule with MyAdder {
    val in1 = IO(Input(UInt(8.W)))
    val in2 = IO(Input(UInt(8.W)))
    val out = IO(Output(UInt(8.W)))
    out := in1 - in2
}

class X[T <: BaseModule with MyAdder](genT: => T) extends Module {
    val io = IO(new Bundle {
        val in1 = Input(UInt(8.W))
        val in2 = Input(UInt(8.W))
        val out = Output(UInt(8.W))
    })
    val subMod = Module(genT)
    io.out := subMod.out
    subMod.in1 := io.in1
    subMod.in2 := io.in2
}

println(ChiselStage.emitSystemVerilog(new X(new Mod1)))
println(ChiselStage.emitSystemVerilog(new X(new Mod2)))
```

Output:

```scala mdoc:verilog
ChiselStage.emitSystemVerilog(new X(new Mod1))
ChiselStage.emitSystemVerilog(new X(new Mod2))
```
