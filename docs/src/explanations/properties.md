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
class Example extends RawModule {
  // An Int Property type port.
  val myPort = IO(Input(Property[Int]()))
}
```
