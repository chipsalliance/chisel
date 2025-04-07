---
layout: docs
title:  "Probes"
section: "chisel3"
---

# Probes

_Probes_ are a way to encode a _reference_ to hardware that will be later
referred to by-name.  Mechanistically, probes are a way to genereate
SystemVerilog which includes hierarchical names (see: Section 23.6 of the
SystemVerilog 2023 specification).

Probes are typically used to expose a "verification interface" to a unit for
debugging, testing, or inspection without adding ports to the final hardware.
When combined with [layers](layers), they may be "layer-colored" and optionally
exist in a design based on Verilog compilation-time decisions.

:::warning

Probes are _not_ shadow dataflow.  They are _not_ a mechanism to connect
arbitrary hardware while avoiding ports.  They are closer to "references" in a
traditional programming language, except that they have extra restrictions.
Namely, a probe will eventually be accessed by-name and that name must, at its
access site, resolve to the probed value unambiguously.

:::

## Overview

There are two kinds of probes based on the type of access that a user wants to
have to a piece of hardware.  A _read probe_ allows for read-only access to
hardware.  A _read--write probe_ allows for both read and write access.

Read probes are typically used for passive verification (e.g., assertions or
monitors) or debugging (e.g., building an architectural debug "view" of a
microarchitecture).  Read--write probes are typically used for more active
verification (e.g., injecting faults to test fault recovery mechanisms or as a
means of closing difficult to reach coverage).

APIs for working with probes are in the `chisel3.probe` package.

### Read Probes

To create a read probe of a hardware value, use the `ProbeValue` API.  To create
a read probe _type_, use the `Probe` API.  Probes are legal types for ports and
wires, but not for stateful elements (e.g., registers or memories).

:::note

It may be surprising the a probe is a legal type for a wire.  However, wires in
Chisel behave more like variables than they do like "hardware wires" (or Verilog
net types).  With this view, it is natural that a probe (a reference) may be
passed through a variable.

:::

Probes are different from normal Chisel hardware types.  Whereas normal Chisel
hardware may be connected to multiple times with the last connection "winning"
via so-called "last-connect semantics", probe types may only be _defined_
exactly once.  The API to define a probe is, unsurprisingly, called `define`.
This is used to "forward" a probe up through the hierarchy, e.g., to define a
probe port with the probed value of a wire.

For convenience, you may alternatively use the standard Chisel connection
operators which will, under-the-hood, use the `define` operator automatically
for you.

To read the value of a probe type, use the `read` API.

The following example shows a circuit that uses all of the APIs introduced
previously.  Both `define` and standard Chisel connection operators are shown.
Careful use of `dontTouch` is used to prevent optimization across probes so that
the output is not trivially simple.

```scala mdoc:silent
import chisel3._
import chisel3.probe.{Probe, ProbeValue, define, read}

class Bar extends RawModule {
  val a_port = IO(Probe(Bool()))
  val b_port = IO(Probe(Bool()))

  private val a = dontTouch(WireInit(Bool(), true.B))
  private val a_probe = ProbeValue(a)
  define(a_port, a_probe)
  b_port :<= a_probe
}

class Foo extends RawModule {

  private val bar = Module(new Bar)

  private val a_read = dontTouch(WireInit(read(bar.a_port)))
  private val b_read = dontTouch(WireInit(read(bar.b_port)))
}

```

The SystemVerilog for the above circuit is shown below:

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification",
    "-enable-layers=Verification.Assert",
    "-enable-layers=Verification.Assume",
    "-enable-layers=Verification.Cover"
  )
)
```

There are several things that are worth highlighting in the above SystemVerilog:

1. The wires `a_read` and `b_read` are driven with _hierarchical names_ that
   reach into module `Bar`.  There are _no ports_ created on module `Bar`.  This
   is intended to support design verification use cases where certain signals
   are made available from `Bar` via probe ports that are then used to, e.g.,
   connect to assertions, monitors, or verification intellectual property (IP).
   If hardware ports were used this would change the interface of the design
   unfavorably.

2. Observability, via probes, is not free.  While the above circuit is contrived
   in its simplicity, if hardware is probed, that may limit the ability of the
   compiler to optimize that hardware.  Read probes are generally more amenable
   to optimization than read--write probes.  However, they still have effects.

### Read--write Probes

To create a read--write probe of a hardware value, use the `RWProbeValue` API.
To create a read--write probe _type_, use the `RWProbe` API.  As with read
probes, read--write probes are legal types for ports and wires, but not for
stateful elements (e.g., registers or memories).

As with read probes, read--write probes forward references using the `define`
API or the standard Chisel connection operators.

A read--write probe can be read using the same `read` API that is used for read
probes.  Multiple different operations are provided for writing to a read--write
probe.  The `force` and `forceInitial` APIs are used to overwrite the value of
read--write probed hardware.  The `release` and `releaseInitial` APIs are used
to stop overwriting a read--write probed hardware value.

:::note

All writing of read--write probes is done through APIs which lower to System
Verilog `force`/`release` statements (see: Section 10.6 of the SystemVerilog
2023 specification).  It is intentionally not possible to use normal Chisel
connects to write to read--write probes.  Put differently, read--write probes do
_not_ participate in last-connect semantics.

:::

The following example shows a circuit that uses all of the APIs introduced
previously.  Both `define` and standard Chisel connection operators are shown.
Careful use of `dontTouch` is used to prevent optimization across probes so that
the output is not trivially simple.

```scala mdoc:reset:silent
import chisel3._
import chisel3.probe.{RWProbe, RWProbeValue, force, forceInitial, read, release, releaseInitial}

class Bar extends RawModule {
  val a_port = IO(RWProbe(Bool()))
  val b_port = IO(RWProbe(UInt(8.W)))

  private val a = WireInit(Bool(), true.B)
  a_port :<= RWProbeValue(a)

  private val b = WireInit(UInt(8.W), 0.U)
  b_port :<= RWProbeValue(b)
}

class Foo extends Module {
  val cond = IO(Input(Bool()))

  private val bar = Module(new Bar)

  // Example usage of forceInitial/releaseInitial:
  forceInitial(bar.a_port, false.B)
  releaseInitial(bar.a_port)

  // Example usage of force/release:
  when (cond) {
    force(bar.b_port, 42.U)
  }.otherwise {
    release(bar.b_port)
  }

  // The read API may still be used:
  private val a_read = dontTouch(WireInit(read(bar.a_port)))
}
```

The SystemVerilog for the above circuit is shown below:

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  Array("--throw-on-first-error"),
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification",
    "-enable-layers=Verification.Assert",
    "-enable-layers=Verification.Assume",
    "-enable-layers=Verification.Cover"
  )
)
```

Several things are worth commenting on in the above SystemVerilog:

1. Writability is very invasive.  In order to compile a write probe, all
   optimizations on its target must be blocked and any optimizations _through_
   the target are not possible.  This is because any writes to a read--write
   probe must affect downstream users.

2. The APIs for writing to read--write probes (e.g., `force`) are extremely
   low-level and very tightly coupled to SystemVerilog.  Take great care when
   using these APIs and validating that the resulting SystemVerilog does what
   you want.

:::warning

Not all simulators correctly implement force and release as described in the
SystemVerilog spec!  Be careful when using read--write probes.  You may need to
use a SystemVerilog-compliant simulator.

:::

## Verilog ABI

Earlier examples only show probes being used internal to a circuit.  However,
probes also compile to SystemVerilog in such a way that they are usable external
to the circuit.

Consider the following example circuit.  In this, an internal register's value
is exposed via a read probe.

```scala mdoc:reset:silent
import chisel3._
import chisel3.probe.{Probe, ProbeValue}

class Foo extends Module {

  val d = IO(Input(UInt(32.W)))
  val q = IO(Output(chiselTypeOf(d)))
  val r_probe = IO(Output(Probe(chiselTypeOf(d))))

  private val r = Reg(chiselTypeOf(d))

  q :<= r

  r_probe :<= ProbeValue(r)
}
```

The SystemVerilog for the above circuit is shown below:

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  Array("--throw-on-first-error"),
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification",
    "-enable-layers=Verification.Assert",
    "-enable-layers=Verification.Assume",
    "-enable-layers=Verification.Cover"
  )
)
```

As part of the compilation, this is guaranteed to produce an additional file for
each public module with a specific filename: `ref_<module-name>.sv`.  In this
file, there will be one SystemVerilog text macro definition for each probe port
of that public module.  The define will have a text macro name derived from the
module name and the probe port name: `ref_<module-name>_<probe-name>`.

Using this ABI, the module may be instantiated elsewhere (e.g., by a
SystemVerilog testbench) and its probed internals accessed.

:::info

For the exact definition of the port lowering ABI for probes, see the [FIRRTL
ABI
Specification](https://github.com/chipsalliance/firrtl-spec/releases/latest/download/abi.pdf).

:::

## Layer-colored Probes

Probes are allowed to be layer-colored.  I.e., this is a mechanism to declare
that a probe's existence is contingent on a specific layer being enabled.  To
declare a probe as being layer-colored, the `Probe` or `RWProbe` type takes an
optional argument indicating what the layer coloring is.  The following example
decalres two probe ports with different layer colors:

```scala mdoc:reset:silent
import chisel3._
import chisel3.layer.{Layer, LayerConfig}
import chisel3.probe.{Probe, ProbeValue}

object A extends Layer(LayerConfig.Extract())
object B extends Layer(LayerConfig.Extract())

class Foo extends Module {
  val a = IO(Output(Probe(Bool(), A)))
  val b = IO(Output(Probe(UInt(8.W), B)))
}
```

For more information on layer-colored probes see [the appropriate subsection of
the layers documentation](layers#layer-colored-probes-and-wires).

## Why Input Probes are Not Allowed

Input probes (of either read or read--write kind) are disallowed.  This is an
intentional decision that stems from requirements of both what probes are and
how probes can be compiled to SystemVerilog.

First, probes are references.  They refer to hardware which exists somewhere
else.  They are not hardware wires.  They are not "shadow" ports.  They do not
represent "shadow" dataflow.

Second, a probe always comes with two pieces: the actual probed hardware and the
operation which uses the reference to the probed hardware.  The operation that
uses the probe must, at its specific location, be able to refer unambiguously to
the probed hardware.  As the example below will show, this is problematic with
input probes.

Consider the following illegal Chisel which uses hypothetical input probes:

``` scala
import chisel3._
import chisel3.probe.{Probe, ProbeValue, read}

module Baz extends RawModule {
  val probe = IO(Input(Probe(Bool())))

  val b = WireInit(read(probe))
}

module Bar extends RawModule {
  val probe = IO(Input(Probe(Bool())))

  val baz = Module(new Baz)
  baz.probe :<= probe

}

module Foo extends RawModule {

  val w = Wire(Bool())

  val bar = Module(new Bar)
  bar.probe :<= ProbeValue(w)
}
```

This could be compiled to the following SystemVerilog:

``` verilog
module Baz();

  wire b = Foo.a;

endmodule

module Bar();

  Baz baz();

endmodule

module Foo();

  wire a;

  Bar bar();

endmodule
```

SystemVerilog provides an algorithm for resolving _upwards_ hierarchical names
(see: Section 23.8 of the SystemVerilog 2023 specification).  This works by
looking in the current scope for a match for the root of the name (`Foo`) and if
it fails, it moves up one level and tries to look aagin.  This then repeats
until a name is found (or errors if the top of the circuit is reached).
However, this algorithm places harsh naming constraints on intermediary modules.
E.g., in the example above, no name `Foo` can exist in `Baz` or in an
_intervening_ modules between `Baz` and `Foo`.  This can easily run afoul of
names which cannot be changed, e.g., public modules or public module ports.

Additionally, any use of a hierarchical name that resolves upwards means that
the module that uses that upwards reference is limited in its ability to be
freely instantiated.  In the circuit above, `Baz` is singly instantiated.
However, if `Baz` was multiply instantiated, it could be given two different
input probes.  This would mean that `Baz` could _not_ be compiled to a single
Verilog module.  It must be duplicated for each unique hierarchical name that it
contains.  This can have cascading duplication effects where parent modules,
their parents, etc. must be duplicated.  The unpredictability of this is not
viewed as tolerable by users.

Both of these constraints (the constraints on names in intevening modules and
duplication to resolve hierarchical names) make the use of input probes
problematic.  While they could be compiled, the results will be unpredictable
and difficult for a user to debug when things go wrong.

Due to these problems, input probes were rejected as a design point and are not
planned to be implemented.

## BoringUtils

Probes are an intentionally a low-level API.  E.g., if a design needs to expose
a probe port, it may need to add probe ports to all intervening modules between
it and a probed value.

For a more flexible API, consider using `chisel3.util.experimental.BoringUtils`.
This provides higher-level APIs that automatically create probe ports for the
user:

- `rwTap`: creates a read--write probe of a signal and routes it to the call
  site
- `tap`: creates a read probe of a signal and routes it to the call site
- `tapAndRead`: `creates a read probe of a signal, routes it to the call site,
  and reads it (converts from probe to real hardware)

E.g., consider the original example shown for read probes.  This can be
rewritten using `BoringUtils` to be more terse:

```scala mdoc:reset:silent
import chisel3._
import chisel3.util.experimental.BoringUtils

class Bar extends RawModule {
  val a = dontTouch(WireInit(Bool(), true.B))
}

class Foo extends RawModule {

  private val bar = Module(new Bar)

  private val a_read = dontTouch(WireInit(BoringUtils.tapAndRead(bar.a)))
}

```

The SystemVerilog for the above circuit is shown below:

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification",
    "-enable-layers=Verification.Assert",
    "-enable-layers=Verification.Assume",
    "-enable-layers=Verification.Cover"
  )
)
```

In order to do this, it requires that the tapped target is public from Scala's
perspective.

:::note

`BoringUtils` is only suitable for use _within_ a compilation unit.
Additionally, excessive use of `BoringUtils` can result in very confusing
hardware generators where the port-level interfaces are unpredictable.

:::

If a `BoringUtils` API is used in a situation which would create an input probe,
it will instead create a non-probe input port.

## Type Cloning and Probes

Chisel internally treats probes as not full types, but as type modifiers.  This
is similar to its treatment of direction (e.g., input or output) and constness.
For this reason, care should be taken when trying to "clone" a probe type and an
understanding of what different APIs do with probes is necessary.  Each API and
its behavior with probes is described below:

- `cloneType`: this will _not_ propagate probe information
- `chiselTypeClone`: this will propagate probe information
- `chiselTypeOf`: this will propagate probe information, but can only be run on
  real hardware
