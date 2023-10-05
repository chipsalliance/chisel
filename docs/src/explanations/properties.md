---
layout: docs
title:  "Properties"
section: "chisel3"
---

# Properties

Chisel *properties* represent information about the design that is not hardware.
This is useful to capture domain-specific knowledge and design intent alongside
the hardware description within the same generator.

## Property Types

The core primitive for using properties is the `Property` type.

`Property` types work similarly to the other Chisel
[Data Types](../explanations/data-types), but rather than specifying the type of
values held in state elements or flowing through wires in the circuit,
properties never flow through or affect the generated hardware. Instead, they
flow through the hierarchy as ports that can be connected.

What makes `Property` types useful is their ability to express non-hardware
information that is present in the generated hierarchy, and can be composed to
create domain-specific data models that are tightly coupled to the design. An
input port with `Property` type represents a part of the data model that must be
supplied when its module is instantiated. An output port with `Property` type
represents a part of the data model that may be accessed when its module is
instantiated. As the complete design is generated, an arbitrary data model can
be generated alongside it.

The following are legal `Property` types:

* `Property[Int]`
* `Property[Long]`
* `Property[BigInt]`
* `Property[String]`
* `Property[Boolean]`
* `Property[Seq[A]]` (where `A` is itself a `Property`)

## Using Properties

The `Property` functionality can be used with the following imports:

```scala mdoc:silent
import chisel3._
import chisel3.properties.Property
```

The subsections below show example uses of `Property` types in various Chisel
constructs.

### Property Ports

The legal `Property` types may be used in ports. For example:

```scala mdoc:silent
class PortsExample extends RawModule {
  // An Int Property type port.
  val myPort = IO(Input(Property[Int]()))
}
```

### Property Connections

The legal `Property` types may be connected using the `:=` operator. For
example, an input `Property` type port may be connected to an output `Property`
type port:

```scala mdoc:silent
class ConnectExample extends RawModule {
  val inPort = IO(Input(Property[Int]()))
  val outPort = IO(Output(Property[Int]()))
  outPort := inPort
}
```

Connections are only supported between the same `Property` type. For example, a
`Property[Int]` may only be connected to a `Property[Int]`. This is enforced by
the Scala compiler.

### Property Values

The legal `Property` types may be used to construct values by applying the
`Property` object to a value of the `Property` type. For example, a
`Property` value may be connected to an output `Property` type port:

```scala mdoc:silent
class LiteralExample extends RawModule {
  val outPort = IO(Output(Property[Int]()))
  outPort := Property(123)
}
```

### Property Sequences

Similarly to the primitive `Property` types, sequences of `Properties` may also be
for creating ports and values and they may also be connected:

```scala mdoc:silent
class SequenceExample extends RawModule {
  val inPort = IO(Input(Property[Int]()))
  val outPort1 = IO(Output(Property[Seq[Int]]()))
  val outPort2 = IO(Output(Property[Seq[Int]]()))
  // A Seq of literals can by turned into a Property
  outPort1 := Property(Seq(123, 456))
  // Property ports and literals can be mixed together into a Seq
  outPort2 := Property(Seq(inPort, Property(789)))
}
```
