---
layout: docs
title:  "Interfaces and Connections"
section: "chisel3"
---

# Interfaces & Connections

For more sophisticated modules it is often useful to define and instantiate interface classes while defining the IO for a module. First and foremost, interface classes promote reuse allowing users to capture once and for all common interfaces in a useful form.

Secondly, interfaces allow users to dramatically reduce wiring by supporting bulk connections between producer and consumer modules. Finally, users can make changes in large interfaces in one place reducing the number of updates required when adding or removing pieces of the interface.

Note that Chisel has some built-in standard interface which should be used whenever possible for interoperability (e.g. Decoupled).

## Ports: Subclasses & Nesting

As we saw earlier, users can define their own interfaces by defining a class that subclasses Bundle. For example, a user could define a simple link for hand-shaking data as follows:

```scala mdoc:invisible
import chisel3._
import circt.stage.ChiselStage
```

```scala mdoc:silent
class SimpleLink extends Bundle {
  val data = Output(UInt(16.W))
  val valid = Output(Bool())
}
```

We can then extend SimpleLink by adding parity bits using bundle inheritance:
```scala mdoc:silent
class PLink extends SimpleLink {
  val parity = Output(UInt(5.W))
}
```
In general, users can organize their interfaces into hierarchies using inheritance.

From there we can define a filter interface by nesting two PLinks into a new FilterIO bundle:
```scala mdoc:silent
class FilterIO extends Bundle {
  val x = Flipped(new PLink)
  val y = new PLink
}
```
where flip recursively changes the direction of a bundle, changing input to output and output to input.

We can now define a filter by defining a filter class extending module:
```scala mdoc:silent
class Filter extends Module {
  val io = IO(new FilterIO)
  // ...
}
```
where the io field contains FilterIO.

## Bundle Vectors

Beyond single elements, vectors of elements form richer hierarchical interfaces. For example, in order to create a crossbar with a vector of inputs, producing a vector of outputs, and selected by a UInt input, we utilize the Vec constructor:
```scala mdoc:silent
import chisel3.util.log2Ceil
class CrossbarIo(n: Int) extends Bundle {
  val in = Vec(n, Flipped(new PLink))
  val sel = Input(UInt(log2Ceil(n).W))
  val out = Vec(n, new PLink)
}
```
where Vec takes a size as the first argument and a block returning a port as the second argument.

## Bulk Connections
Once we have a defined Interface, we can connect to it via a [`MonoConnect`](https://www.chisel-lang.org/api/latest/chisel3/Data.html#:=) operator (`:=`) or [`BiConnect`](https://www.chisel-lang.org/api/latest/chisel3/Data.html#%3C%3E) operator (`<>`).


### `MonoConnect` Algorithm
`MonoConnect.connect`, or `:=`, executes a mono-directional connection element-wise.

Note that this isn't commutative. There is an explicit source and sink
already determined before this function is called.

The connect operation will recurse down the left Data (with the right Data).
An exception will be thrown if a movement through the left cannot be matched
in the right. The right side is allowed to have extra fields.
Vecs must still be exactly the same size.

Note that the LHS element must be writable so, one of these must hold:
- Is an internal writable node (`Reg` or `Wire`)
- Is an output of the current module
- Is an input of a submodule of the current module

Note that the RHS element must be readable so, one of these must hold:
- Is an internal readable node (`Reg`, `Wire`, `Op`)
- Is a literal
- Is a port of the current module or submodule of the current module


### `BiConnect` Algorithm
`BiConnect.connect`, or `<>`, executes a bidirectional connection element-wise. Note that the arguments are left and right (not source and sink) so the intent is for the operation to be commutative. The connect operation will recurse down the left `Data` (with the right `Data`). An exception will be thrown if a movement through the left cannot be matched in the right, or if the right side has extra fields.

> Note: We highly encourage new code to be written with the [`Connectable` Operators](https://www.chisel-lang.org/chisel3/docs/explanations/connectable.html) rather than the `<>` operator.

Using the biconnect `<>` operator, we can now compose two filters into a filter block as follows:
```scala mdoc:silent
class Block extends Module {
  val io = IO(new FilterIO)
  val f1 = Module(new Filter)
  val f2 = Module(new Filter)
  f1.io.x <> io.x
  f1.io.y <> f2.io.x
  f2.io.y <> io.y
}
```

The bidirectional bulk connection operator `<>` connects leaf ports of the same name to each other. The Scala types of the Bundles are not required to match. If one named signal is missing from either side, Chisel will give an error such as in the following example:

```scala mdoc:silent

class NotReallyAFilterIO extends Bundle {
  val x = Flipped(new PLink)
  val y = new PLink
  val z = Output(new Bool())
}
class Block2 extends Module {
  val io1 = IO(new FilterIO)
  val io2 = IO(Flipped(new NotReallyAFilterIO))

  io1 <> io2
}
```
Below we can see the resulting error for this example:
```scala mdoc:crash
ChiselStage.emitSystemVerilog(new Block2)
```
Bidirectional connections should only be used with **directioned elements** (like IOs), e.g. connecting two wires isn't supported since Chisel can't necessarily figure out the directions automatically.
For example, putting two temporary wires and connecting them here will not work, even though the directions could be known from the endpoints:

```scala mdoc:silent

class BlockWithTemporaryWires extends Module {
  val io = IO(new FilterIO)
  val f1 = Module(new Filter)
  val f2 = Module(new Filter)
  f1.io.x <> io.x
 val tmp1 = Wire(new FilterIO)
 val tmp2 = Wire(new FilterIO)
  f1.io.y <> tmp1
  tmp1 <> tmp2
  tmp2 <> f2.io.x
  f2.io.y <> io.y
}

```
Below we can see the resulting error for this example:
```scala mdoc:crash
ChiselStage.emitSystemVerilog(new BlockWithTemporaryWires)
```
For more details and information, see [Deep Dive into Connection Operators](connection-operators)

NOTE: When using `Chisel._` (compatibility mode) instead of `chisel3._`, the `:=` operator works in a bidirectional fashion similar to `<>`, but not exactly the same.

## The standard ready-valid interface (ReadyValidIO / Decoupled)

Chisel provides a standard interface for [ready-valid interfaces](http://inst.eecs.berkeley.edu/~cs150/Documents/Interfaces.pdf).
A ready-valid interface consists of a `ready` signal, a `valid` signal, and some data stored in `bits`.
The `ready` bit indicates that a consumer is *ready* to consume data.
The `valid` bit indicates that a producer has *valid* data on `bits`.
When both `ready` and `valid` are asserted, a data transfer from the producer to the consumer takes place.
A convenience method `fire` is provided that is asserted if both `ready` and `valid` are asserted.

Usually, we use the utility function [`Decoupled()`](https://chisel.eecs.berkeley.edu/api/latest/chisel3/util/Decoupled$.html) to turn any type into a ready-valid interface rather than directly using [ReadyValidIO](http://chisel.eecs.berkeley.edu/api/latest/chisel3/util/ReadyValidIO.html).

* `Decoupled(...)` creates a producer / output ready-valid interface (i.e. bits is an output).
* `Flipped(Decoupled(...))` creates a consumer / input ready-valid interface (i.e. bits is an input).

Take a look at the following example Chisel code to better understand exactly what is generated:

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.Decoupled

/**
  * Using Decoupled(...) creates a producer interface.
  * i.e. it has bits as an output.
  * This produces the following ports:
  *   input         io_readyValid_ready,
  *   output        io_readyValid_valid,
  *   output [31:0] io_readyValid_bits
  */
class ProducingData extends Module {
  val io = IO(new Bundle {
    val readyValid = Decoupled(UInt(32.W))
  })
  // do something with io.readyValid.ready
  io.readyValid.valid := true.B
  io.readyValid.bits := 5.U
}

/**
  * Using Flipped(Decoupled(...)) creates a consumer interface.
  * i.e. it has bits as an input.
  * This produces the following ports:
  *   output        io_readyValid_ready,
  *   input         io_readyValid_valid,
  *   input  [31:0] io_readyValid_bits
  */
class ConsumingData extends Module {
  val io = IO(new Bundle {
    val readyValid = Flipped(Decoupled(UInt(32.W)))
  })
  io.readyValid.ready := false.B
  // do something with io.readyValid.valid
  // do something with io.readyValid.bits
}
```

`DecoupledIO` is a ready-valid interface with the *convention* that there are no guarantees placed on deasserting `ready` or `valid` or on the stability of `bits`.
That means `ready` and `valid` can also be deasserted without a data transfer.

`IrrevocableIO` is a ready-valid interface with the *convention* that the value of `bits` will not change while `valid` is asserted and `ready` is deasserted.
Also, the consumer shall keep `ready` asserted after a cycle where `ready` was high and `valid` was low.
Note that the *irrevocable* constraint *is only a convention* and cannot be enforced by the interface.
Chisel does not automatically generate checkers or assertions to enforce the *irrevocable* convention.
