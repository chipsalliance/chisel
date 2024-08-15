---
layout: docs
title:  "Properties"
section: "chisel3"
---

# Properties

Chisel *properties* represent information about the design that is not hardware.
This is useful to capture domain-specific knowledge and design intent alongside
the hardware description within the same generator.

:::warning

Properties are under active development and are not yet considered stable.

:::

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

### Property Expressions

Expressions can be built out of `Property` values for certain `Property` types.
This is useful for expressing design intent that is parameterized by input
`Property` values.

#### Integer Arithmetic

The integral `Property` types, like `Property[Int]`, `Property[Long]` and 
`Property[BigInt]`, can be used to build arithmetic expressions in terms of
`Property` values.

In the following example, an output `address` port of `Property[Int]` type is
computed as the addition of an `offset` `Property[Int]` value relative to an 
input `base` `Property[Int]` value.

```scala mdoc:silent
class IntegerArithmeticExample extends RawModule {
  val base = IO(Input(Property[Int]()))
  val address = IO(Output(Property[Int]()))
  val offset = Property(1024)
  address := base + offset
}
```

The following table lists the possible arithmetic operators that are supported
on integral `Property` typed values.

| Operation | Description                                                                         |
| --------- | -----------                                                                         |
| `+`       | Perform addition as defined by FIRRTL spec section Integer Add Operation            |
| `*`       | Perform multiplication as defined by FIRRTL spec section Integer Multiply Operation |
| `>>`      | Perform shift right as defined by FIRRTL spec section Integer Shift Right Operation |

#### Sequence Operations

The sequence `Property` types, like `Property[Seq[Int]]` support some basic
operations to create new sequences from existing sequences.

In the following example, and output `c` port of `Property[Seq[Int]]` type is
computed as the concatenation of the `a` and `b` ports of `Property[Seq[Int]]`
type.

```scala mdoc:silent
class SequenceOperationExample extends RawModule {
  val a = IO(Input(Property[Seq[Int]]()))
  val b = IO(Input(Property[Seq[Int]]()))
  val c = IO(Output(Property[Seq[Int]]()))
  c := a ++ b
}
```

The following table lists the possible sequence operators that are supported on
sequence `Property` typed values.

| Operation | Description                                                                          |
| --------- | -----------                                                                          |
| `++`      | Perform concatenation as defined by FIRRTL spec section List Concatenation Operation |

### Classes and Objects

Classes and Objects are to `Property` types what modules and instances are to
hardware types. That is, they provide a means to declare hierarchies through
which `Property` typed values flow. `Class` declares a hierarchical container
with input and output `Property` ports, and a body that contains `Property`
connections and `Object`s. `Object`s represent the instantiation of a `Class`,
which requires any input `Property` ports to be assigned, and allows any output
`Property` ports to be read.

This allows domain-specific data models to be built using the basic primitives
of an object-oriented programming language, and embedded directly in the
instance graph Chisel is constructing. Intuitively, inputs to a `Class` are like
constructor arguments, which must be supplied to create an `Object`. Similarly,
outputs from a `Class` are like fields, which may be accessed from an `Object`.
This separation allows `Class` declarations to abstract over any `Object`s
created in their body--from the outside, the inputs must be supplied and only
the outputs may be accessed.

The graphs represented by `Class` declarations and `Object` instantiations
coexist within the hardware instance graph. `Object` instances can exist
within hardware modules, providing domain-specific information, but hardware
instances cannot exist within `Class` declarations.

`Object`s can be referenced, and references to `Object`s are a special kind of
`Property[ClassType]` type. This allows the data model captured by `Class`
declarations and `Object` instances to form arbitrary graphs.

To understand how `Object` graphs are represented, and can ultimately be
queried, consider how the hardware instance graph is elaborated. To build the
`Object` graph, we first pick an entrypoint module to start elaboration. The
elaboration process works according to the Verilog spec's definition of
elaboration--instances of modules and `Object`s are instantiated in-memory,
with connections to their inputs and outputs. Inputs are supplied, and outputs
may be read. After elaboration completes, the `Object` graph is exposed in terms
of the output ports, which may contain any `Property` types, including
references to `Object`s.

To illustrate how these pieces come together, consider the following examples:

```scala mdoc:silent
import chisel3.properties.Class
import chisel3.experimental.hierarchy.{instantiable, public, Definition, Instance}

// An abstract description of a CSR, represented as a Class.
@instantiable
class CSRDescription extends Class {
  // An output Property indicating the CSR name.
  val identifier = IO(Output(Property[String]()))
  // An output Property describing the CSR.
  val description = IO(Output(Property[String]()))
  // An output Property indicating the CSR width.
  val width = IO(Output(Property[Int]()))

  // Input Properties to be passed to Objects representing instances of the Class.
  @public val identifierIn = IO(Input(Property[String]()))
  @public val descriptionIn = IO(Input(Property[String]()))
  @public val widthIn = IO(Input(Property[Int]()))

  // Simply connect the inputs to the outputs to expose the values.
  identifier := identifierIn
  description := descriptionIn
  width := widthIn
}
```

The `CSRDescription` is a `Class` that represents domain-specific information
about a CSR. This uses `@instantiable` and `@public` so the `Class` can work
with the `Definition` and `Instance` APIs.

The readable fields we want to expose on `Object`s of the `CSRDescription` class
are a string identifier, a string description, and an integer bitwidth, so these
are output `Property` type ports on the `Class`.

To capture concrete values at each `Object` instantiation, we have corresponding
input `Property` type ports, which are connected directly to the outputs. This
is how we would represent something like a Scala `case class` using `Class`.

```scala mdoc:silent
// A hardware module representing a CSR and its description.
class CSRModule(
  csrDescDef:     Definition[CSRDescription],
  width:          Int,
  identifierStr:  String,
  descriptionStr: String)
    extends Module {
  override def desiredName = identifierStr

  // Create a hardware port for the CSR value.
  val value = IO(Output(UInt(width.W)))

  // Create a property port for a reference to the CSR description object.
  val description = IO(Output(csrDescDef.getPropertyType))

  // Instantiate a CSR description object, and connect its input properties.
  val csrDescription = Instance(csrDescDef)
  csrDescription.identifierIn := Property(identifierStr)
  csrDescription.descriptionIn := Property(descriptionStr)
  csrDescription.widthIn := Property(width)

  // Create a register for the hardware CSR. A real implementation would be more involved.
  val csr = RegInit(0.U(width.W))

  // Assign the CSR value to the hardware port.
  value := csr

  // Assign a reference to the CSR description object to the property port.
  description := csrDescription.getPropertyReference
}
```

The `CSRModule` is a `Module` that represents the (dummy) hardware for a CSR, as
well as a `CSRDescription`. Using a `Definition` of a `CSRDescription`, an
`Object` is created and its inputs supplied from the `CSRModule` constructor
arguments. Then, a reference to the `Object` is connected to the `CSRModule`
output, so the reference will be exposed to the outside.

```scala mdoc:silent
// The entrypoint module.
class Top extends Module {
  // Create a Definition for the CSRDescription Class.
  val csrDescDef = Definition(new CSRDescription)

  // Get the CSRDescription ClassType.
  val csrDescType = csrDescDef.getClassType

  // Create a property port to collect all the CSRDescription object references.
  val descriptions = IO(Output(Property[Seq[csrDescType.Type]]()))

  // Instantiate a couple CSR modules.
  val mcycle = Module(new CSRModule(csrDescDef, 64, "mcycle", "Machine cycle counter."))
  val minstret = Module(new CSRModule(csrDescDef, 64, "minstret", "Machine instructions-retired counter."))

  // Assign references to the CSR description objects to the property port.
  descriptions := Property(Seq(mcycle.description.as(csrDescType), minstret.description.as(csrDescType)))
}
```

The `Top` module represents the entrypoint. It creates the `Definition` of the
`CSRDescription`, and creates some `CSRModule`s. It then takes the description
references, collects them into a list, and outputs the list so it will be
exposed to the outside.

While it is not required to use the `Definition` API to define a `Class`, this
is the "safe" API, with support in Chisel for working with `Definition`s and
`Instance`s of a `Class`. There is also an "unsafe" API. See `DynamicObject` for
more information.

To illustrate what this example generates, here is a listing of the FIRRTL:

```
FIRRTL version 4.0.0
circuit Top :
  class CSRDescription :
    output identifier : String
    output description : String
    output width : Integer
    input identifierIn : String
    input descriptionIn : String
    input widthIn : Integer

    propassign identifier, identifierIn
    propassign description, descriptionIn
    propassign width, widthIn

  module mcycle :
    input clock : Clock
    input reset : Reset
    output value : UInt<64>
    output description : Inst<CSRDescription>

    object csrDescription of CSRDescription
    propassign csrDescription.identifierIn, String("mcycle")
    propassign csrDescription.descriptionIn, String("Machine cycle counter.")
    propassign csrDescription.widthIn, Integer(64)
    regreset csr : UInt<64>, clock, reset, UInt<64>(0h0)
    connect value, csr
    propassign description, csrDescription

  module minstret :
    input clock : Clock
    input reset : Reset
    output value : UInt<64>
    output description : Inst<CSRDescription>

    object csrDescription of CSRDescription
    propassign csrDescription.identifierIn, String("minstret")
    propassign csrDescription.descriptionIn, String("Machine instructions-retired counter.")
    propassign csrDescription.widthIn, Integer(64)
    regreset csr : UInt<64>, clock, reset, UInt<64>(0h0)
    connect value, csr
    propassign description, csrDescription

  public module Top :
    input clock : Clock
    input reset : UInt<1>
    output descriptions : List<Inst<CSRDescription>>

    inst mcycle of mcycle
    connect mcycle.clock, clock
    connect mcycle.reset, reset
    inst minstret of minstret
    connect minstret.clock, clock
    connect minstret.reset, reset
    propassign descriptions, List<Inst<CSRDescription>>(mcycle.description, minstret.description)
```

To understand the `Object` graph that is constructed, we will consider an
entrypoint to elaboration, and then show a hypothetical JSON representation of
the `Object` graph. The details of how we go from IR to an `Object` graph are
outside the scope of this document, and implemented by related tools.

If we elaborate `Top`, the `descriptions` output `Property` is our entrypoint to
the `Object` graph. Within it, there are two `Object`s, the `CSRDescription`s of
the `mcycle` and `minstret` modules:

```json mdoc:silent
{
  "descriptions": [
    {
      "identifier": "mcycle",
      "description": "Machine cycle counter.",
      "width": 64
    },
    {
      "identifier": "minstret",
      "description": "Machine instructions-retired counter.",
      "width": 64
    }
  ]
}
```

If instead, we elaborate one of the `CSRModule`s, for example, `minstret`, the
`description` output `Property` is our entrypoint to the `Object` graph, which
contains the single `CSRDescription` object:

```json mdoc:silent
{
  "description": {
    "identifier": "minstret",
    "description": "Machine instructions-retired counter.",
    "width": 64
  }
}
```

In this way, the output `Property` ports, `Object` references, and choice of
elaboration entrypoint allow us to view the `Object` graph representing the
domain-specific data model from different points in the hierarchy.
