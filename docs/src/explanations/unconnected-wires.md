---
layout: docs
title:  "Unconnected Wires"
section: "chisel3"
---

# Unconnected Wires

The Invalidate API [(#645)](https://github.com/freechipsproject/chisel3/pull/645) adds support to Chisel
for reporting unconnected wires as errors.

Prior to this pull request, Chisel automatically generated a firrtl `is invalid` for `Module IO()`, and each `Wire()` definition.
This made it difficult to detect cases where output signals were never driven.
Chisel now supports a `DontCare` element, which may be connected to an output signal, indicating that that signal is intentionally not driven.
Unless a signal is driven by hardware or connected to a `DontCare`, Firrtl will complain with a "not fully initialized" error.

### API

Output signals may be connected to DontCare, generating a `is invalid` when the corresponding firrtl is emitted.

```scala mdoc:invisible
import chisel3._
```
```scala mdoc:silent

class Out extends Bundle { 
  val debug = Bool()
  val debugOption = Bool()
}
val io = new Bundle { val out = new Out }
```

```scala mdoc:compile-only
io.out.debug := true.B
io.out.debugOption := DontCare
```

This indicates that the signal `io.out.debugOption` is intentionally not driven and firrtl should not issue a "not fully initialized"
error for this signal.

This can be applied to aggregates as well as individual signals:
```scala mdoc:invisible
import chisel3._
```
```scala mdoc:silent
import chisel3._
class ModWithVec extends Module {
  // ...
  val nElements = 5
  val io = IO(new Bundle {
    val outs = Output(Vec(nElements, Bool()))
  })
  io.outs <> DontCare
  // ...
}

class TrivialInterface extends Bundle {
  val in  = Input(Bool())
  val out = Output(Bool())
}

class ModWithTrivalInterface extends Module {
  // ...
  val io = IO(new TrivialInterface)
  io <> DontCare
  // ...
}
```

### Determining the unconnected element

I have an interface with 42 wires.
Which one of them is unconnected?

The firrtl error message should contain something like:
```bash
firrtl.passes.CheckInitialization$RefNotInitializedException:  @[:@6.4] : [module Router]  Reference io is not fully initialized.
   @[Decoupled.scala 38:19:@48.12] : node _GEN_23 = mux(and(UInt<1>("h1"), eq(UInt<2>("h3"), _T_84)), _GEN_2, VOID) @[Decoupled.scala 38:19:@48.12]
   @[Router.scala 78:30:@44.10] : node _GEN_36 = mux(_GEN_0.ready, _GEN_23, VOID) @[Router.scala 78:30:@44.10]
   @[Router.scala 75:26:@39.8] : node _GEN_54 = mux(io.in.valid, _GEN_36, VOID) @[Router.scala 75:26:@39.8]
   @[Router.scala 70:50:@27.6] : node _GEN_76 = mux(io.load_routing_table_request.valid, VOID, _GEN_54) @[Router.scala 70:50:@27.6]
   @[Router.scala 65:85:@19.4] : node _GEN_102 = mux(_T_62, VOID, _GEN_76) @[Router.scala 65:85:@19.4]
   : io.outs[3].bits.body <= _GEN_102
```
The first line is the initial error report.
Successive lines, indented and beginning with source line information indicate connections involving the problematic signal.
Unfortunately, if these are `when` conditions involving muxes, they may be difficult to decipher.
The last line of the group, indented and beginning with a `:` should indicate the uninitialized signal component.
This example (from the [Router tutorial](https://github.com/ucb-bar/chisel-tutorial/blob/release/src/main/scala/examples/Router.scala))
was produced when the output queue bits were not initialized.
The old code was:
```scala
  io.outs.foreach { out => out.noenq() }
```
which initialized the queue's `valid` bit, but did not initialize the actual output values.
The fix was:
```scala
  io.outs.foreach { out =>
    out.bits := 0.U.asTypeOf(out.bits)
    out.noenq()
  }
```
