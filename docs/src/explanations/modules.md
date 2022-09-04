---
layout: docs
title:  "Modules"
section: "chisel3"
---

# Modules

Chisel *modules* are very similar to Verilog *modules* in
defining a hierarchical structure in the generated circuit.

The hierarchical module namespace is accessible in downstream tools
to aid in debugging and physical layout.  A user-defined module is
defined as a *class* which:

 - inherits from `Module`,
 - contains at least one interface wrapped in a Module's `IO()` method (traditionally stored in a port field named ```io```), and
 - wires together subcircuits in its constructor.

As an example, consider defining your own two-input multiplexer as a
module:
```scala mdoc:silent
import chisel3._
class Mux2IO extends Bundle {
  val sel = Input(UInt(1.W))
  val in0 = Input(UInt(1.W))
  val in1 = Input(UInt(1.W))
  val out = Output(UInt(1.W))
}

class Mux2 extends Module {
  val io = IO(new Mux2IO)
  io.out := (io.sel & io.in1) | (~io.sel & io.in0)
}
```

The wiring interface to a module is a collection of ports in the
form of a ```Bundle```.  The interface to the module is defined
through a field named ```io```.  For ```Mux2```, ```io``` is
defined as a bundle with four fields, one for each multiplexer port.

The ```:=``` assignment operator, used here in the body of the
definition, is a special operator in Chisel that wires the input of
left-hand side to the output of the right-hand side.

### Module Hierarchy

We can now construct circuit hierarchies, where we build larger modules out
of smaller sub-modules.  For example, we can build a 4-input
multiplexer module in terms of the ```Mux2``` module by wiring
together three 2-input multiplexers:

```scala mdoc:silent
class Mux4IO extends Bundle {
  val in0 = Input(UInt(1.W))
  val in1 = Input(UInt(1.W))
  val in2 = Input(UInt(1.W))
  val in3 = Input(UInt(1.W))
  val sel = Input(UInt(2.W))
  val out = Output(UInt(1.W))
}
class Mux4 extends Module {
  val io = IO(new Mux4IO)

  val m0 = Module(new Mux2)
  m0.io.sel := io.sel(0)
  m0.io.in0 := io.in0
  m0.io.in1 := io.in1

  val m1 = Module(new Mux2)
  m1.io.sel := io.sel(0)
  m1.io.in0 := io.in2
  m1.io.in1 := io.in3

  val m3 = Module(new Mux2)
  m3.io.sel := io.sel(1)
  m3.io.in0 := m0.io.out
  m3.io.in1 := m1.io.out

  io.out := m3.io.out
}
```

We again define the module interface as ```io``` and wire up the
inputs and outputs.  In this case, we create three ```Mux2```
children modules, using the ```Module``` constructor function and
the Scala ```new``` keyword to create a
new object.  We then wire them up to one another and to the ports of
the ```Mux4``` interface.

Note: Chisel `Module`s have an implicit clock (called `clock`) and
an implicit reset (called `reset`). To create modules without implicit
clock and reset, Chisel provides `RawModule`.

### `RawModule`

A `RawModule` is a module that **does not provide an implicit clock and reset.**
This can be useful when interfacing a Chisel module with a design that expects
a specific naming convention for clock or reset.

Then we can use it in place of *Module* usage :
```scala mdoc:silent
import chisel3.{RawModule, withClockAndReset}

class Foo extends Module {
  val io = IO(new Bundle{
    val a = Input(Bool())
    val b = Output(Bool())
  })
  io.b := !io.a
}

class FooWrapper extends RawModule {
  val a_i  = IO(Input(Bool()))
  val b_o  = IO(Output(Bool()))
  val clk  = IO(Input(Clock()))
  val rstn = IO(Input(Bool()))

  val foo = withClockAndReset(clk, !rstn){ Module(new Foo) }

  foo.io.a := a_i
  b_o := foo.io.b
}
```

In the example above, the `RawModule` is used to change the reset polarity
of module `SlaveSpi`. Indeed, the reset is active high by default in Chisel
modules, then using `withClockAndReset(clock, !rstn)` we can use an active low
reset in the entire design.

The clock is just wired as is, but if needed, `RawModule` can be used in
conjunction with `BlackBox` to connect a differential clock input for example.
