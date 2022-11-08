---

layout: docs

title:  "Connectable Operators"

section: "chisel3"

---

## Terminology

- "Chisel type" - a `Data` that is not bound to hardware (more details[here](chisel-type-vs-scala-type)).
  - E.g. `UInt(3.W)`, `new Bundle {..}`, `Vec(3, SInt(2.W))` are all Chisel types
- `Aggregate` - a Chisel type or component that contains other Chisel types or components (i.e. `Vec`, `Record`, or `Bundle`)
- `Element` - a Chisel type or component that does not contain other Chisel types or components (e.g. `UInt`, `SInt`, `Clock`, `Bool` etc.)
- "component" - a `Data` that is bound to hardware (`IO`, `Reg`, `Wire`, etc.)
  - E.g. `Wire(UInt(3.W))` is a component, whose Chisel type is `UInt(3.W)`
- "member" - a Chisel type or component, or any of its children (could be an `Aggregate` or an `Element`)
  - E.g. `Vec(3, UInt(2.W))(0)` is a member of the parent `Vec` Chisel type
  - E.g. `Wire(Vec(3, UInt(2.W)))(0)` is a member of the parent `Wire` component
  - E.g. `IO(Decoupled(Bool)).ready` is a member of the parent `IO` component
  

For more details about these Scala types vs Chisel types, please read 

## Overview

The `Connectable` operators are the standard way to connect Chisel hardware components to one another.

> Note: For descriptions of the semantics for the previous operators, see [`Connection Operators`](connection-operators).

All connection operators require the two hardware components (consumer and producer) to be Chisel type-equivalent (matching bundle field names and types (`Record` vs `Vector` vs `Element`), vector sizes, `Element` types (UInt/SInt/Bool/Clock etc)). Use `DataMirror.checkTypeEquivalence` to check this property.

The one exception to the type-equivalence rule is using the `WaivedData` mechanism, detailed at [section](#waived-data) at the end of this document.

Aggregate (`Record`, `Vec`, `Bundle`) Chisel types can include data members which are flipped relative to one another. Due to this, there are many desired connection behaviors between two Chisel components. The following are the Chisel connection operators:
 - `c := p` (mono-direction): assigns all p members to c; requires c & p to not have any flipped members
 - `c :#= p` (coercing mono-direction): assigns all p members to c; regardless of alignment
 - `c :<= p` (aligned-direction); assigns all aligned (non-flipped) c members from p
 - `c :>= p` (flipped-direction); assigns all flipped p members from c
 - `c :<>= p` (bi-direction operator); assigns all aligned c members from p; all flipped p members from c

You may be seeing these random symbols in the operator and going "what the heck are these?!?". Well, it turns out that the characters are consistent between operators and self-describe the semantics of each operator:
 - `:` always indicates the consumer, or left-hand-side, of the operator.
 - `=` always indicates the producer, or right-hand-side, of the operator.
   - Hence, `c := p` connects a consumer (`c`) and a producer (`p`).
 - `<` always indicates that some members will be driven producer-to-consumer, or right-to-left.
   - Hence, `c :<= p` drives members in producer (`p`) to members consumer (`c`).
 - `>` always indicates that some signals will be driven consumer-to-producer, or left-to-right.
   - Hence, `c :>= p` drives members in consumer (`c`) to members producer (`p`).
   - Hence, `c :<>= p` both drives members from `p` to `c` and from `c` to `p`.
 - `#` always indicates to ignore member alignment and to drive producer-to-consumer.
   - Hence, `c :#= p` always drives members from `p` to `c` ignoring direction.


## Alignment: Flipped vs Aligned

A member's alignment is a relative property: a member is aligned/flipped relative to another member of the same component or Chisel type. Hence, one must always say whether a member is flipped/aligned *with respect to (w.r.t)* another member of that type (parent, sibling, child etc.).

We use the following example of a non-nested bundle `Parent` to let us state all of the alignment relationships between members of `p`.

```scala mdoc:silent
import chisel3._
class Parent extends Bundle {
  val alignedChild = UInt(32.W)
  val flippedChild = Flipped(UInt(32.W))
}
class MyModule0 extends Module {
  val p = Wire(new Parent)
}
```

First, every member is always aligned with themselves:
 - `p` is aligned w.r.t `p`
 - `p.alignedChild` is aligned w.r.t `p.alignedChild`
 - `p.flippedChild` is aligned w.r.t `p.flippedChild`

Next, we list all parent/child relationships. Because the `flippedChild` field is `Flipped`, it changes its aligment relative to its parent. 
 - `p` is aligned w.r.t `p.alignedChild`
 - `p` is flipped w.r.t `p.flippedChild`

Finally, we can list all sibling relationships:
 - `p.alignedChild` is flipped w.r.t `p.flippedChild`

The next example has a nested bundle `GrandParent` who instantiates an aligned `Parent` field and flipped `Parent` field.

```scala mdoc:silent
import chisel3._
class GrandParent extends Bundle {
  val alignedParent = new Parent()
  val flippedParent = Flipped(new Parent())
}
class MyModule1 extends Module {
  val g = Wire(new GrandParent)
}
```

Consider the following alignements between grandparent and grandchildren. An odd number of flips indicate a flipped relationship; even numbers of flips indicate an aligned relationship.
 - `g` is aligned w.r.t `g.flippedParent.flippedChild`
 - `g` is aligned w.r.t `g.alignedParent.alignedChild`
 - `g` is flipped w.r.t `g.flippedParent.alignedChild`
 - `g` is flipped w.r.t `g.alignedParent.flippedChild`

Consider the following alignment relationships starting from `g.alignedParent` and `g.flippedParent`. *Note that whether `g.alignedParent` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.alignedParent` and `g.alignedParent.alignedChild` because alignment is only relative to the two members in question!*:
 - `g.alignedParent` is aligned w.r.t. `g.alignedParent.alignedChild`
 - `g.flippedParent` is aligned w.r.t. `g.flippedParent.alignedChild`
 - `g.alignedParent` is flipped w.r.t. `g.alignedParent.flippedChild`
 - `g.flippedParent` is flipped w.r.t. `g.flippedParent.flippedChild`

In summary, a member is aligned or flipped w.r.t. another member of the hardware component. This means that the type of the consumer/producer is the only information needed to determine the behavior of any operator. *Whether the consumer/producer is a member of a larger bundle is irrelevant; you ONLY need to know the type of the consumer/producer*.

## Input/Output

`Input(gen)`/`Output(gen)` are coercing operators. They perform two functions: (1) create a new Chisel type that has all flips removed from all recursive children members but structurally equivalent to `gen`, and (2) apply `Flipped` if `Input`, keep aligned (do nothing) if `Output`. E.g. if we imagine a function called `cloneChiselTypeButStripAllFlips`, then `Input(gen)` is equivalent to `Flipped(cloneChiselTypeButStripAllFlips(gen))`.

Note that if `gen` is a non-aggregate, then `Input(nonAggregateGen)` is equivalent to `Flipped(nonAggregateGen)`.

> Future work will refactor how these primitives are exposed to the user to make Chisel's type system more intuitive. See [https://github.com/chipsalliance/chisel3/issues/2643].

With this in mind, we can consider the following examples and detail relative alignments of members.

First, we can use a similar example to `Parent` but use `Input/Output` instead of `Flipped`.
Because `alignedChild` and `flippedChild` are non-aggregates, `Input` is basically just a `Flipped` and thus the alignments are unchanged compared to the previous `Parent` example.

```scala mdoc:silent
import chisel3._
class ParentWithOutputInput extends Bundle {
  val alignedCoerced = Output(UInt(32.W)) // Equivalent to just UInt(32.W)
  val flippedCoerced = Input(UInt(32.W))  // Equivalent to Flipped(UInt(32.W))
}
class MyModule2 extends Module {
  val p = Wire(new ParentWithOutputInput)
}
```

The aligments are the same as the previous `Parent` example:
 - `p` is aligned w.r.t `p`
 - `p.alignedCoerced` is aligned w.r.t `p.alignedCoerced`
 - `p.flippedCoerced` is aligned w.r.t `p.flippedCoerced`
 - `p` is aligned w.r.t `p.alignedCoerced`
 - `p` is flipped w.r.t `p.flippedCoerced`
 - `p.alignedCoerced` is flipped w.r.t `p.flippedCoerced`

The next example has a nested bundle `GrandParent` who instantiates an `Output` `ParentWithOutputInput` field and an `Input` `ParentWithOutputInput` field.

```scala mdoc:silent
import chisel3._
class GrandParentWithOutputInput extends Bundle {
  val alignedCoerced = Output(new ParentWithOutputInput())
  val flippedCoerced = Input(new ParentWithOutputInput())
}
class MyModule3 extends Module {
  val g = Wire(new GrandParentWithOutputInput)
}
```

Remember that `Output(gen)/Input(gen)` recursively strips the `Flipped` of any recursive children.
This makes every member of `gen` aligned with every other member of `gen`.

Consider the following alignments between grandparent and grandchildren. Because `alignedCoerced` and `flippedCoerced` are aligned with all their recursive members, they are fully aligned. Thus, only their alignment to `g` influences grandchildren alignment:
 - `g` is aligned w.r.t `g.alignedCoerced.alignedChild`
 - `g` is aligned w.r.t `g.alignedCoerced.flippedChild`
 - `g` is flipped w.r.t `g.flippedCoerced.alignedChild`
 - `g` is flipped w.r.t `g.flippedCoerced.flippedChild`

Consider the following alignment relationships starting from `g.alignedCoerced` and `g.flippedCoerced`. *Note that whether `g.alignedCoerced` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.alignedCoerced` and `g.alignedCoerced.alignedChild` or `g.alignedCoerced.flippedChild` because alignment is only relative to the two members in question! However, because alignment is coerced, everything is aligned between `g.alignedCoerced`/`g.flippedAligned` and their children*:
 - `g.alignedCoerced` is aligned w.r.t. `g.alignedCoerced.alignedChild`
 - `g.flippedCoerced` is aligned w.r.t. `g.alignedCoerced.flippedChild`
 - `g.alignedCoerced` is aligned w.r.t. `g.flippedCoerced.alignedChild`
 - `g.flippedCoerced` is aligned w.r.t. `g.flippedCoerced.flippedChild`

In summary, `Input(gen)` and `Output(gen)` recursively coerce children alignment, as well as dictate `gen`'s alignment to its parent bundle (if it exists).

## Connecting components with fully aligned members

For simple connections where all members are aligned (non-flipped) w.r.t. one another, use `:=`:


```scala mdoc:silent
import chisel3._
class FullyAlignedBundle extends Bundle {
  val a = Bool()
  val b = Bool()
}
class Example0 extends RawModule {
  val incoming = IO(Flipped(new FullyAlignedBundle))
  val outgoing = IO(new FullyAlignedBundle)
  outgoing := incoming
}
```

This generates the following Verilog, where each member of `incoming` drives every member of `outgoing`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example0())
```

> You may be thinking "Wait, I'm confused! Isn't foo flipped and bar aligned?" -- Noo! Whether foo is aligned with bar makes no sense; remember, you only evaluate alignment between members of the same component or Chisel type. Because components are always aligned to themselves, `bar` is aligned to `bar`, and `foo` is aligned to `foo`, there is no problem. Their relative flippedness to anything else is irrelevant.

## Connections for components with members of mixed-alignment

Aggregate Chisel types can include data members which are flipped relative to one another; in the example below, `alignedChild` and `flippedChild` are aligned/flipped relative to `MixedAlignmentBundle`.

```scala mdoc:silent
import chisel3._
class MixedAlignmentBundle extends Bundle {
  val alignedChild = Bool()
  val flippedChild = Flipped(Bool())
}
```

Due to this, there are many desired connection behaviors between two Chisel components. First we will introduce the most common Chisel connection operator, `:<>=`, useful for connecting components with members of mixed-alignments, then take a moment to investigate a common source of confusion between port-direction and connection-direction. Then, we will explore the remainder of the the Chisel connection operators.


### Bi-direction connection operator (:<>=)

For connections where you want 'bulk-connect-like-semantics' where the aligned members are driven producer-to-consumer and flipped members are driven consumer-to-producer, use `:<>=`.

```scala mdoc:silent
class Example1 extends RawModule {
  val incoming = IO(Flipped(new MixedAlignmentBundle))
  val outgoing = IO(new MixedAlignmentBundle)
  outgoing :<>= incoming
}
```

This generates the following Verilog, where the aligned members are driven `incoming` to `outgoing` and flipped members are driven `outgoing` to `incoming`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example1())
```

### Port-Direction Computation versus Connection-Direction Computation

A common question is if you use a mixed-alignment connection (such as `:<>=`) to connect submembers of parent components, does the alignment of the submember to their parent affect anything? The answer is no, because *alignment is always computed relative to what is being connected to, and members are always aligned with themselves.*

In the following example connecting `incoming.alignedChild` to `outgoing.alignedChild`, whether `incoming.alignedChild` is aligned with `incoming` is irrelevant because the `:<>=` only computes alignment relative to the thing being connected to, and `incoming.alignedChild` is aligned with `incoming.alignedChild`.

```scala mdoc:silent
class Example1a extends RawModule {
  val incoming = IO(Flipped(new MixedAlignmentBundle))
  val outgoing = IO(new MixedAlignmentBundle)
  outgoing.alignedChild :<>= incoming.alignedChild // whether incoming.alignedChild is aligned/flipped to incoming is IRRELEVANT to what gets connected with :<>=
}
```

While `incoming.flippedChild`'s alignment with `incoming` does not affect our operators, it does influence whether `incoming.flippedChild` is an output or input port of my module.
A common source of confusion is to mistake the process for determining whether `incoming.flippedChild` will resolve to a verilog `output`/`input` (the port-direction computation) with the process for determining how `:<>=` drives what with what (the connection-direction computation).
While both processes consider relative alignment, they are distinct.

The port-direction computation always computes alignment relative to the component marked with `IO`. An `IO(Flipped(gen))` is an input port, and any member of `gen` that is aligned/flipped with `gen` is an input/output port. An `IO(gen)` is an output port, and any member of `gen` that is aligned/flipped with `gen` is an output/input port.

The connection-direction computation always computes alignment based on the explicit consumer/producer referenced for the connection. If one connects `incoming :<>= outgoing`, alignments are computed based on `incoming` and `outgoing`. If one connects `incoming.alignedChild :<>= outgoing.alignedChild`, then alignments are computed based on `incoming.alignedChild` and `outgoing.alignedChild` (and the alignment of `incoming` to `incoming.alignedChild` is irrelevant).

This means that users can try to assign to input ports of their module! If I write `x :<>= y`, and `x` is an input to the current module, then that is what the connection is trying to do. However, because input ports are not assignable from within the current module, Chisel will throw an error. This is the same error a user would get using a mono-directioned operator: `x := y` will throw the same error if `x` is an input to the current module. *Whether a component is assignable is irrelevant to the semantics of any connection operator assigning to it.*

In summary, the port-direction computation is relative to the root marked `IO`, but connection-direction computation is relative to the consumer/producer that the connection is doing. This has the positive property that connection semantics are solely based on the Chisel types of the consumer/producer (nothing more, nothing less).

### Aligned connection operator (:<=)

For connections where you want the aligned-half of 'bulk-connect-like-semantics' where the aligned members are driven producer-to-consumer and flipped members are ignored, use `:<=` (the "aligned connection").

```scala mdoc:silent
class Example2 extends RawModule {
  val incoming = IO(Flipped(new MixedAlignmentBundle))
  val outgoing = IO(new MixedAlignmentBundle)
  incoming.flippedChild := DontCare // Otherwise FIRRTL throws an uninitialization error
  outgoing :<= incoming
}
```

This generates the following Verilog, where the aligned members are driven `incoming` to `outgoing` and flipped members are ignored:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example2())
```

### Flipped connection operator (:>=)

For connections where you want the flipped-half of 'bulk-connect-like-semantics' where the aligned members are ignored and flipped members are assigned consumer-to-producer, use `:<=` (the "flipped connection", or "backpressure connection").

```scala mdoc:silent
class Example3 extends RawModule {
  val incoming = IO(Flipped(new MixedAlignmentBundle))
  val outgoing = IO(new MixedAlignmentBundle)
  outgoing.alignedChild := DontCare // Otherwise FIRRTL throws an uninitialization error
  outgoing :>= incoming
}
```

This generates the following Verilog, where the aligned members are ignore and the flipped members are driven `outgoing` to `incoming`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example3())
```

> Note: Astute observers will realize that semantically `c :<>= p` is exactly equivalent to `c :<= p` followed by `c :>= p`.

### Coercing mono-direction connection operator (:#=)

For connections where you want to every producer member to always drive every consumer member, regardless of alignment, use `:#=` (the "coercion connection"). This operator is useful for initializing wires whose types contain members of mixed alignment.

```scala mdoc:silent
import chisel3.experimental.BundleLiterals._
class Example4 extends RawModule {
  val w = Wire(new MixedAlignmentBundle)
  dontTouch(w) // So we see it in the output verilog
  w :#= (new MixedAlignmentBundle).Lit(_.alignedChild -> true.B, _.flippedChild -> true.B)
}
```

This generates the following Verilog, where all members are driven from the literal to `w`, regardless of alignment:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example4())
```

> Note: Astute observers will realize that semantically `c :#= p` is exactly equivalent to `c :<= p` followed by `p :>= c` (note `p` and `c` switched places in the second connection).

Another use case for `:#=` is for connecting a mixed-directional bundle to a fully-aligned monitor.

```scala mdoc:silent
import chisel3.experimental.BundleLiterals._
class Example4b extends RawModule {
  val monitor = IO(Output(new MixedAlignmentBundle))
  val w = Wire(new MixedAlignmentBundle)
  dontTouch(w) // So we see it in the output verilog
  w :#= DontCare
  monitor :#= w
}
```

This generates the following Verilog, where all members are driven from the literal to `w`, regardless of alignment:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example4b())
```
## Waived Data

It is not uncommon for a user to want to connect Chisel components which are not type equivalent. For example, a user may want to hook up the `ready`/`valid` members of a `ReadyValidIO` to a `DecoupledIO`, but
because the `bits` member is not present in both, our operators would reject a connection.

`WaivedData` is the mechanism to specialize connection operator behavior in these scenarios. For any addition member which is not present in the other component being connected to, they can be explicitly waived from the operator to be ignored, rather than trigger an error.

### Connecting sub-types to super-types by waiving extra members

In the following example, we can use `:<>=` to connect a `MyReadyValid` to a `MyDecoupled` by waiving the `bits` member.

```scala mdoc:silent
class MyReadyValid extends Bundle {
  val valid = Bool()
  val ready = Flipped(Bool())
}
class MyDecoupled extends MyReadyValid {
  val bits = UInt(32.W)
}
class Example5 extends RawModule {
  val in  = IO(Flipped(new MyDecoupled()))
  val out = IO(new MyReadyValid())
  out :<>= in.waive(_.bits)
}
```

This generates the following Verilog, where `ready` and `valid` are connected, and `bits` is ignored:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example5())
```

### Connecting types with optional members

In the following example, we can use `:<>=` and `waive` to connect two `MyDecoupledOpts`'s, where only one has a `bits` member.

```scala mdoc:silent
class MyDecoupledOpt(hasBits: Boolean) extends Bundle {
  val valid = Bool()
  val ready = Flipped(Bool())
  val bits = if (hasBits) Some(UInt(32.W)) else None
}
class Example6 extends RawModule {
  val in  = IO(Flipped(new MyDecoupledOpt(true)))
  val out = IO(new MyDecoupledOpt(false))
  out :<>= in.waive(_.bits.get) // We can know to call .get because we can inspect in.bits.isEmpty
}
```

This generates the following Verilog, where `ready` and `valid` are connected, and `bits` is ignored:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example6())
```

### Connecting different sub-types

Note that the connection operator requires the `consumer` and `producer` to be the same Scala type to encourage capturing more information statically, but they can always be cast to `Data` or another common supertype prior to connecting.

In the following example, we can use `:<>=` and `waiveAs` to connect two different sub-types of `MyReadyValid`.

```scala mdoc:silent
class HasBits extends MyReadyValid {
  val bits = UInt(32.W)
}
class HasEcho extends MyReadyValid {
  val echo = Flipped(UInt(32.W))
}
class Example7 extends RawModule {
  val in  = IO(Flipped(new HasBits()))
  val out = IO(new HasEcho())
  out.waiveAs[MyReadyValid](_.echo) :<>= in.waiveAs[MyReadyValid](_.bits)
}
```

This generates the following Verilog, where `ready` and `valid` are connected, and `bits` and `echo` are ignored:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example7())
```

### Defaults with waived connections


A not uncommon usecase is to try to connect two Records; for matching members, they should be connected, but for unmatched members, they should be assigned a default value. To accomplish this, use the other operators to initialize all Record members, then use `:<>=` with `waiveAll` to connect only the matching members.


```scala mdoc:silent
// TODO: Fix this example - AutoCloneType is broken??
import scala.collection.immutable.SeqMap
import chisel3.experimental.AutoCloneType
class MyRecord(elems: () => SeqMap[String, Data]) extends Record with AutoCloneType {
  val elements = elems()
}
class Example8 extends RawModule {
  val abType = new MyRecord(() => SeqMap("a" -> Bool(), "b" -> Flipped(Bool())))
  val bcType = new MyRecord(() => SeqMap("b" -> Flipped(Bool()), "c" -> Bool()))

  val p = Wire(abType)
  val c = Wire(bcType)

  dontTouch(p) // So it doesn't get constant-propped away for the example
  dontTouch(c) // So it doesn't get constant-propped away for the example

  p :#= abType.Lit(_.elements("a") -> true.B, _.elements("b") -> true.B)
  c :#= bcType.Lit(_.elements("b") -> true.B, _.elements("c") -> true.B)

  c.waiveAll :<>= p.waiveAll
}
```

This generates the following Verilog, where the `b` member is driven from `c` to `p`, and `a` and `c` members are initialized to default values:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example8())
```


### Always ignore extra members (partial connection operator)


## Comparison to Chisel.<>, chisel3.<>, chisel3.:=

## WaivedData vs Dataview

TODO: Perhaps this is better served in a cookbook mdoc?

Options available to user:

 - manually bursting out individual fields
 - Waived data
 - .viewAsSuperType
 - static cast to supertype (fields still have to match, but importantly different than viewAsSuperType in the output)
 - dataview to get it to be the right type
 - something else?