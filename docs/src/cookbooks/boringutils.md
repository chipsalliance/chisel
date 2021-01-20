---
layout: docs
title:  "Boring Utils Cookbook"
section: "chisel3"
---

# Boring Utils Cookbook

```scala mdoc
import chisel3._
import chisel3.util.experimental._
import chisel3.stage.ChiselStage
```

Chisel has some experimental utilities for generating synthesizable cross module references that "bore" through the
hierarchy. The underlying cross module connects are handled by FIRRTL's Wiring Transform.

Consider the following example where you want to connect a component in one module to a component in another. Module
`Constant` has a wire tied to `42` and `Expect` will assert unless connected to `42`:
```scala mdoc
import chisel3._
class Constant extends MultiIOModule {
  val x = Wire(UInt(6.W))
  x := 42.U
}
class Expect extends MultiIOModule {
  val y = Wire(UInt(6.W))
  y := 0.U
  // This assertion will fail unless we bore!
  chisel3.assert(y === 42.U, "y should be 42 in module Expect")
}
```

We can then drive `y` with `x` using [[BoringUtils]] without modifying the Chisel IO of `Constant`, `Expect`, or
modules that may instantiate them. There are two approaches to do this:

1. Hierarchical boring using [[BoringUtils.bore]]

2. Non-hierarchical boring using [[BoringUtils.addSink]]/[[BoringUtils.addSource]]

### Hierarchical Boring

Hierarchical boring involves driving one sink signal from a source signal, where the sink and source are not in the same module.
Below, module `Top` contains an instance of `Constant`, called `constant`, and an instance of `Expect` called `expect`.
Using [[BoringUtils.bore]], we can drive `expect.y` from `constant.x`.

```scala mdoc
class Top extends MultiIOModule {
  val constant = Module(new Constant)
  val expect = Module(new Expect)
  BoringUtils.bore(constant.x, Seq(expect.y))
}
println(ChiselStage.emitVerilog(new Top))
```

In addition, you can specify additional modules to route the connection through, where the strings "first" and "second"
are the names of the intermediate wires instantiated in the modules we route through.

```scala mdoc
class Dummy extends MultiIOModule { }
class TopThroughModules extends MultiIOModule {
  val constant = Module(new Constant)
  val dummy1 = Module(new Dummy)
  val dummy2 = Module(new Dummy)
  val expect = Module(new Expect)
  BoringUtils.bore(constant.x, Seq(expect.y), Seq(("first", dummy1), ("second", dummy2)))
}
println(ChiselStage.emitVerilog(new TopThroughModules))
```

### Non-hierarchical Boring

Non-hierarchical boring involves connections from sources to sinks that cannot both be referenced
within the same module. In this example, we build up a source and sink pair called `uniqueId`.
`x` is described as a source and associated with the name `uniqueId`, and `y` is described as a sink with the same name. This is
equivalent to the hierarchical boring example above, but requires no modifications to `Top`.
 
```scala mdoc:reset
import chisel3._
import chisel3.util.experimental._
import chisel3.stage.ChiselStage
class Constant extends MultiIOModule {
  val x = Wire(UInt(6.W))
  x := 42.U
  BoringUtils.addSource(x, "uniqueId")
}
class Expect extends MultiIOModule {
  val y = Wire(UInt(6.W))
  y := 0.U
  // This assertion will fail unless we find a way to drive y!
  chisel3.assert(y === 42.U, "y should be 42 in module Expect")
  BoringUtils.addSink(y, "uniqueId")
}
class Top extends MultiIOModule {
  val constant = Module(new Constant)
  val expect = Module(new Expect)
}
println(ChiselStage.emitVerilog(new Top))
```

### Additional Comments

Both hierarchical and non-hierarchical boring emit FIRRTL annotations that describe sources and sinks. These are
matched by a `name` key that indicates they should be wired together. Hierarchical boring safely generates this name
automatically. Non-hierarchical boring unsafely relies on user input to generate this name. Use of non-hierarchical
naming may result in naming conflicts that the user must handle.

The automatic generation of hierarchical names relies on a global, mutable namespace. This is currently persistent
across circuit elaborations, which means multiple circuit elaborations in the same Scala program execution would share
the same namespace.

You can bore a one-to-many relationship from source to sinks, but not a many-to-many or many-to-one.

