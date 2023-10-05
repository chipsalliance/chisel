---
layout: docs
title:  "Connectable Operators"
section: "chisel3"
---

## Table of Contents
 * [Terminology](#terminology)
 * [Overview](#overview)
 * [Alignment: Flipped vs Aligned](#alignment-flipped-vs-aligned)
 * [Input/Output](#inputoutput)
 * [Connecting components with fully aligned members](#connecting-components-with-fully-aligned-members)
   * [Mono-direction connection operator (:=)](#mono-direction-connection-operator-)
 * [Connecting components with mixed alignment members](#connecting-components-with-mixed-alignment-members)
   * [Bi-direction connection operator (:<>=)](#bi-direction-connection-operator-)
   * [Port-Direction Computation versus Connection-Direction Computation](#port-direction-computation-versus-connection-direction-computation)
   * [Aligned connection operator (:<=)](#aligned-connection-operator-)
   * [Flipped connection operator (:>=)](#flipped-connection-operator-)
   * [Coercing mono-direction connection operator (:#=)](#coercing-mono-direction-connection-operator-)
 * [Connectable](#connectable)
   * [Connecting Records](#connecting-records)
   * [Defaults with waived connections](#defaults-with-waived-connections)
   * [Connecting types with optional members](#connecting-types-with-optional-members)
   * [Always ignore extra members (partial connection operator)](#always-ignore-extra-members-partial-connection-operator)
   * [Connecting components with different widths](#connecting-components-with-different-widths)
 * [Techniques for connecting structurally inequivalent Chisel types](#techniques-for-connecting-structurally-inequivalent-chisel-types)
   * [Connecting different sub-types of the same super-type, with colliding names](#connecting-different-sub-types-of-the-same-super-type-with-colliding-names)
   * [Connecting sub-types to super-types by waiving extra members](#connecting-sub-types-to-super-types-by-waiving-extra-members)
   * [Connecting different sub-types](#connecting-different-sub-types)
 * [FAQ](#faq)

## Terminology

 * "Chisel type" - a `Data` that is not bound to hardware, i.e. not a component. (more details [here](chisel-type-vs-scala-type)).
   * E.g. `UInt(3.W)`, `new Bundle {..}`, `Vec(3, SInt(2.W))` are all Chisel types
 * `Aggregate` - a Chisel type or component that contains other Chisel types or components (i.e. `Vec`, `Record`, or `Bundle`)
 * `Element` - a Chisel type or component that does not contain other Chisel types or components (e.g. `UInt`, `SInt`, `Clock`, `Bool` etc.)
 * "component" - a `Data` that is bound to hardware (`IO`, `Reg`, `Wire`, etc.)
   * E.g. `Wire(UInt(3.W))` is a component, whose Chisel type is `UInt(3.W)`
 * "member" - a Chisel type or component, or any of its children (could be an `Aggregate` or an `Element`)
   * E.g. `Vec(3, UInt(2.W))(0)` is a member of the parent `Vec` Chisel type
   * E.g. `Wire(Vec(3, UInt(2.W)))(0)` is a member of the parent `Wire` component
   * E.g. `IO(Decoupled(Bool)).ready` is a member of the parent `IO` component
 * "relative alignment" - whether two members of the same component or Chisel type are aligned/flipped, relative to one another
   * see section [below](#alignment-flipped-vs-aligned) for a detailed definition
 * "structural type check" - Chisel type `A` is structurally equivalent to Chisel type `B` if `A` and `B` have matching bundle field names and types (`Record` vs `Vector` vs `Element`), probe modifiers (probe vs nonprobe), vector sizes, `Element` types (UInt/SInt/Bool/Clock)
   * ignores relative alignment (flippedness)
 * "alignment type check" - a Chisel type `A` matches alignment with another Chisel type `B` if every member of `A`'s relative alignment to `A` is the same as the structurally corresponding member of `B`'s relative alignment to `B`.

## Overview

The `Connectable` operators are the standard way to connect Chisel hardware components to one another.

> Note: For descriptions of the semantics for the previous operators, see [`Connection Operators`](connection-operators).

All connection operators require the two hardware components (consumer and producer) to be structurally type equivalent.

The one exception to the structural type-equivalence rule is using the `Connectable` mechanism, detailed at this [section](#waived-data) towards the end of this document.

Aggregate (`Record`, `Vec`, `Bundle`) Chisel types can include data members which are flipped relative to one another.
Due to this, there are many desired connection behaviors between two Chisel components.
The following are the Chisel connection operators:
 * `c := p` (mono-direction): connects all p members to c; requires c & p to not have any flipped members
 * `c :#= p` (coercing mono-direction): connects all p members to c; regardless of alignment
 * `c :<= p` (aligned-direction): connects all aligned (non-flipped) c members from p
 * `c :>= p` (flipped-direction): connects all flipped p members from c
 * `c :<>= p` (bi-direction operator): connects all aligned c members from p; all flipped p members from c

These operators may appear to be a random collection of symbols; however, the characters are consistent between operators and self-describe the semantics of each operator:
 * `:` always indicates the consumer, or left-hand-side, of the operator.
 * `=` always indicates the producer, or right-hand-side, of the operator.
   * Hence, `c := p` connects a consumer (`c`) and a producer (`p`).
 * `<` always indicates that some members will be driven producer-to-consumer, or right-to-left.
   * Hence, `c :<= p` drives members in producer (`p`) to members in consumer (`c`).
 * `>` always indicates that some signals will be driven consumer-to-producer, or left-to-right.
   * Hence, `c :>= p` drives members in consumer (`c`) to members producer (`p`).
   * Hence, `c :<>= p` both drives members from `p` to `c` and from `c` to `p`.
 * `#` always indicates to ignore member alignment and to drive producer-to-consumer.
   * Hence, `c :#= p` always drives members from `p` to `c` ignoring direction.

> Note: in addition, an operator that ends in `=` has assignment-precendence, which means that `x :<>= y + z` will translate to `x :<>= (y + z)`, rather than `(x :<>= y) + z`.
This was not true of the `<>` operator and was a minor painpoint for users.


## Alignment: Flipped vs Aligned

A member's alignment is a relative property: a member is aligned/flipped relative to another member of the same component or Chisel type.
Hence, one must always say whether a member is flipped/aligned *with respect to (w.r.t)* another member of that type (parent, sibling, child etc.).

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
 * `p` is aligned w.r.t `p`
 * `p.alignedChild` is aligned w.r.t `p.alignedChild`
 * `p.flippedChild` is aligned w.r.t `p.flippedChild`

Next, we list all parent/child relationships.
Because the `flippedChild` field is `Flipped`, it changes its aligment relative to its parent. 
 * `p` is aligned w.r.t `p.alignedChild`
 * `p` is flipped w.r.t `p.flippedChild`

Finally, we can list all sibling relationships:
 * `p.alignedChild` is flipped w.r.t `p.flippedChild`

The next example has a nested bundle `GrandParent` who instantiates an aligned `Parent` field and flipped `Parent` field.

```scala mdoc:silent
import chisel3._
class GrandParent extends Bundle {
  val alignedParent = new Parent
  val flippedParent = Flipped(new Parent)
}
class MyModule1 extends Module {
  val g = Wire(new GrandParent)
}
```

Consider the following alignements between grandparent and grandchildren.
An odd number of flips indicate a flipped relationship; even numbers of flips indicate an aligned relationship.
 * `g` is aligned w.r.t `g.flippedParent.flippedChild`
 * `g` is aligned w.r.t `g.alignedParent.alignedChild`
 * `g` is flipped w.r.t `g.flippedParent.alignedChild`
 * `g` is flipped w.r.t `g.alignedParent.flippedChild`

Consider the following alignment relationships starting from `g.alignedParent` and `g.flippedParent`.
*Note that whether `g.alignedParent` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.alignedParent` and `g.alignedParent.alignedChild` because alignment is only relative to the two members in question!*:
 * `g.alignedParent` is aligned w.r.t. `g.alignedParent.alignedChild`
 * `g.flippedParent` is aligned w.r.t. `g.flippedParent.alignedChild`
 * `g.alignedParent` is flipped w.r.t. `g.alignedParent.flippedChild`
 * `g.flippedParent` is flipped w.r.t. `g.flippedParent.flippedChild`

In summary, a member is aligned or flipped w.r.t. another member of the hardware component.
This means that the type of the consumer/producer is the only information needed to determine the behavior of any operator.
*Whether the consumer/producer is a member of a larger bundle is irrelevant; you ONLY need to know the type of the consumer/producer*.

## Input/Output

`Input(gen)`/`Output(gen)` are coercing operators.
They perform two functions: (1) create a new Chisel type that has all flips removed from all recursive children members (still structurally equivalent to `gen` but no longer alignment type equivalent), and (2) apply `Flipped` if `Input`, keep aligned (do nothing) if `Output`.
E.g. if we imagine a function called `cloneChiselTypeButStripAllFlips`, then `Input(gen)` is structurally and alignment type equivalent to `Flipped(cloneChiselTypeButStripAllFlips(gen))`.

Note that if `gen` is a non-aggregate, then `Input(nonAggregateGen)` is equivalent to `Flipped(nonAggregateGen)`.

> Future work will refactor how these primitives are exposed to the user to make Chisel's type system more intuitive.
See [https://github.com/chipsalliance/chisel3/issues/2643].

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
 * `p` is aligned w.r.t `p`
 * `p.alignedCoerced` is aligned w.r.t `p.alignedCoerced`
 * `p.flippedCoerced` is aligned w.r.t `p.flippedCoerced`
 * `p` is aligned w.r.t `p.alignedCoerced`
 * `p` is flipped w.r.t `p.flippedCoerced`
 * `p.alignedCoerced` is flipped w.r.t `p.flippedCoerced`

The next example has a nested bundle `GrandParent` who instantiates an `Output` `ParentWithOutputInput` field and an `Input` `ParentWithOutputInput` field.

```scala mdoc:silent
import chisel3._
class GrandParentWithOutputInput extends Bundle {
  val alignedCoerced = Output(new ParentWithOutputInput)
  val flippedCoerced = Input(new ParentWithOutputInput)
}
class MyModule3 extends Module {
  val g = Wire(new GrandParentWithOutputInput)
}
```

Remember that `Output(gen)/Input(gen)` recursively strips the `Flipped` of any recursive children.
This makes every member of `gen` aligned with every other member of `gen`.

Consider the following alignments between grandparent and grandchildren.
Because `alignedCoerced` and `flippedCoerced` are aligned with all their recursive members, they are fully aligned.
Thus, only their alignment to `g` influences grandchildren alignment:
 * `g` is aligned w.r.t `g.alignedCoerced.alignedChild`
 * `g` is aligned w.r.t `g.alignedCoerced.flippedChild`
 * `g` is flipped w.r.t `g.flippedCoerced.alignedChild`
 * `g` is flipped w.r.t `g.flippedCoerced.flippedChild`

Consider the following alignment relationships starting from `g.alignedCoerced` and `g.flippedCoerced`.
*Note that whether `g.alignedCoerced` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.alignedCoerced` and `g.alignedCoerced.alignedChild` or `g.alignedCoerced.flippedChild` because alignment is only relative to the two members in question! However, because alignment is coerced, everything is aligned between `g.alignedCoerced`/`g.flippedAligned` and their children*:
 * `g.alignedCoerced` is aligned w.r.t. `g.alignedCoerced.alignedChild`
 * `g.alignedCoerced` is aligned w.r.t. `g.alignedCoerced.flippedChild`
 * `g.flippedCoerced` is aligned w.r.t. `g.flippedCoerced.alignedChild`
 * `g.flippedCoerced` is aligned w.r.t. `g.flippedCoerced.flippedChild`

In summary, `Input(gen)` and `Output(gen)` recursively coerce children alignment, as well as dictate `gen`'s alignment to its parent bundle (if it exists).

## Connecting components with fully aligned members

### Mono-direction connection operator (:=)

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
getVerilogString(new Example0)
```

> You may be thinking "Wait, I'm confused! Isn't `incoming` flipped and `outgoing` aligned?" -- Noo! Whether `incoming` is aligned with `outgoing` makes no sense; remember, you only evaluate alignment between members of the same component or Chisel type.
Because components are always aligned to themselves, `outgoing` is aligned to `outgoing`, and `incoming` is aligned to `incoming`, there is no problem.
Their relative flippedness to anything else is irrelevant.

## Connecting components with mixed alignment members

Aggregate Chisel types can include data members which are flipped relative to one another; in the example below, `alignedChild` and `flippedChild` are aligned/flipped relative to `MixedAlignmentBundle`.

```scala mdoc:silent
import chisel3._
class MixedAlignmentBundle extends Bundle {
  val alignedChild = Bool()
  val flippedChild = Flipped(Bool())
}
```

Due to this, there are many desired connection behaviors between two Chisel components.
First we will introduce the most common Chisel connection operator, `:<>=`, useful for connecting components with members of mixed-alignments, then take a moment to investigate a common source of confusion between port-direction and connection-direction.
Then, we will explore the remainder of the the Chisel connection operators.


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
getVerilogString(new Example1)
```

### Port-Direction Computation versus Connection-Direction Computation

A common question is if you use a mixed-alignment connection (such as `:<>=`) to connect submembers of parent components, does the alignment of the submember to their parent affect anything? The answer is no, because *alignment is always computed relative to what is being connected to, and members are always aligned with themselves.*

In the following example connecting from `incoming.alignedChild` to `outgoing.alignedChild`, whether `incoming.alignedChild` is aligned with `incoming` is irrelevant because the `:<>=` only computes alignment relative to the thing being connected to, and `incoming.alignedChild` is aligned with `incoming.alignedChild`.

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

The port-direction computation always computes alignment relative to the component marked with `IO`.
An `IO(Flipped(gen))` is an incoming port, and any member of `gen` that is aligned/flipped with `gen` is an incoming/outgoing port.
An `IO(gen)` is an outgoing port, and any member of `gen` that is aligned/flipped with `gen` is an outgoing/incoming port.

The connection-direction computation always computes alignment based on the explicit consumer/producer referenced for the connection.
If one connects `incoming :<>= outgoing`, alignments are computed based on `incoming` and `outgoing`.
If one connects `incoming.alignedChild :<>= outgoing.alignedChild`, then alignments are computed based on `incoming.alignedChild` and `outgoing.alignedChild` (and the alignment of `incoming` to `incoming.alignedChild` is irrelevant).

This means that users can try to connect to input ports of their module! If I write `x :<>= y`, and `x` is an input to the current module, then that is what the connection is trying to do.
However, because input ports are not drivable from within the current module, Chisel will throw an error.
This is the same error a user would get using a mono-directioned operator: `x := y` will throw the same error if `x` is an input to the current module.
*Whether a component is drivable is irrelevant to the semantics of any connection operator attempting to drive to it.*

In summary, the port-direction computation is relative to the root marked `IO`, but connection-direction computation is relative to the consumer/producer that the connection is doing.
This has the positive property that connection semantics are solely based on the Chisel structural type and its relative alignments of the consumer/producer (nothing more, nothing less).

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
getVerilogString(new Example2)
```

### Flipped connection operator (:>=)

For connections where you want the flipped-half of 'bulk-connect-like-semantics' where the aligned members are ignored and flipped members are connected consumer-to-producer, use `:>=` (the "flipped connection", or "backpressure connection").

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
getVerilogString(new Example3)
```

> Note: Astute observers will realize that semantically `c :<>= p` is exactly equivalent to `c :<= p` followed by `c :>= p`.

### Coercing mono-direction connection operator (:#=)

For connections where you want to every producer member to always drive every consumer member, regardless of alignment, use `:#=` (the "coercion connection").
This operator is useful for initializing wires whose types contain members of mixed alignment.

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
getVerilogString(new Example4)
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
getVerilogString(new Example4b)
```
## Connectable

It is not uncommon for a user to want to connect Chisel components which are not type equivalent.
For example, a user may want to hook up anonymous `Record` components who may have an intersection of their fields being equivalent, but cannot because they are not structurally equivalent.
Alternatively, one may want to connect two types that have different widths.

`Connectable` is the mechanism to specialize connection operator behavior in these scenarios.
For additional members which are not present in the other component being connected to, or for mismatched widths, or for always excluding a member from being connected too, they can be explicitly called out from the `Connectable` object, rather than trigger an error.

In addition, there are other techniques that can be used to address similar use cases including `.viewAsSuperType`, a static cast to a supertype (e.g. `(x: T)`), or creating a custom dataview.
For a discussion about when to use each technique, please continue [here](#techniques-for-connecting-structurally-inequivalent-chisel-types).

This section demonstrates how `Connectable` specifically can be used in a multitude of scenarios.

### Connecting Records

A not uncommon usecase is to try to connect two Records; for matching members, they should be connected, but for unmatched members, the errors caused due to them being unmatched should be ignored.
To accomplish this, use the other operators to initialize all Record members, then use `:<>=` with `waiveAll` to connect only the matching members.

> Note that none of `.viewAsSuperType`, static casts, nor a custom DataView helps this case because the Scala types are still `Record`.

```scala mdoc:silent
import scala.collection.immutable.SeqMap

class Example9 extends RawModule {
  val abType = new Record { val elements = SeqMap("a" -> Bool(), "b" -> Flipped(Bool())) }
  val bcType = new Record { val elements = SeqMap("b" -> Flipped(Bool()), "c" -> Bool()) }

  val p = IO(Flipped(abType))
  val c = IO(bcType)

  DontCare :>= p
  c :<= DontCare

  c.waive(_.elements("c")):<>= p.waive(_.elements("a"))
}
```

This generates the following Verilog, where `p.b` is driven from `c.b`:

```scala mdoc:verilog
getVerilogString(new Example9)
```

### Defaults with waived connections

Another not uncommon usecase is to try to connect two Records; for matching members, they should be connected, but for unmatched members, *they should be connected a default value*.
To accomplish this, use the other operators to initialize all Record members, then use `:<>=` with `waiveAll` to connect only the matching members.


```scala mdoc:silent
import scala.collection.immutable.SeqMap

class Example10 extends RawModule {
  val abType = new Record { val elements = SeqMap("a" -> Bool(), "b" -> Flipped(Bool())) }
  val bcType = new Record { val elements = SeqMap("b" -> Flipped(Bool()), "c" -> Bool()) }

  val p = Wire(abType)
  val c = Wire(bcType)

  dontTouch(p) // So it doesn't get constant-propped away for the example
  dontTouch(c) // So it doesn't get constant-propped away for the example

  p :#= abType.Lit(_.elements("a") -> true.B, _.elements("b") -> true.B)
  c :#= bcType.Lit(_.elements("b") -> true.B, _.elements("c") -> true.B)

  c.waive(_.elements("c")) :<>= p.waive(_.elements("a"))
}
```

This generates the following Verilog, where `p.b` is driven from `c.b`, and `p.a`, `c.b`, and `c.c` are initialized to default values:

```scala mdoc:verilog
getVerilogString(new Example10)
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
getVerilogString(new Example6)
```

### Always ignore errors caused by extra members (partial connection operator)

The most unsafe connection is to connect only members that are present in both consumer and producer, and ignore all other members.
This is unsafe because this connection will never error on any Chisel types.

To do this, you can use `.waiveAll` and static cast to `Data`:

```scala mdoc:silent
class OnlyA extends Bundle {
  val a = UInt(32.W)
}
class OnlyB extends Bundle {
  val b = UInt(32.W)
}
class Example11 extends RawModule {
  val in  = IO(Flipped(new OnlyA))
  val out = IO(new OnlyB)

  out := DontCare

  (out: Data).waiveAll :<>= (in: Data).waiveAll
}
```

This generates the following Verilog, where nothing is connected:

```scala mdoc:verilog
getVerilogString(new Example11)
```

### Connecting components with different widths

Non-connectable operators implicitly truncate if a component with a larger width is connected to a component with a smaller width.
Connectable operators disallow this implicit truncation behavior and require the driven component to be equal or larger in width that the sourcing component.

If implicit truncation behavior is desired, then `Connectable` provides a `squeeze` mechanism which will allow the connection to continue and implicit trunction to continue.

```scala mdoc:silent
import scala.collection.immutable.SeqMap

class Example14 extends RawModule {
  val p = IO(Flipped(UInt(4.W)))
  val c = IO(UInt(3.W))

  c :<>= p.squeeze
}
```

This generates the following Verilog, where `p` is implicitly truncated prior to driving `c`:

```scala mdoc:verilog
getVerilogString(new Example14)
```

### Excluding members from any operator on a Connectable

If a user wants to always exclude a field from a connect, use the `exclude` mechanism which will never connect the field (as if it didn't exist to the connection).

Note that if a field matches in both producer and consumer, but only one is excluded, the other non-excluded field will still trigger an error; to fix this, use either `waive` or `exclude`.

```scala mdoc:silent
import scala.collection.immutable.SeqMap

class BundleWithSpecialField extends Bundle {
  val foo = UInt(3.W)
  val special = Bool()
}
class Example15 extends RawModule {
  val p = IO(Flipped(new BundleWithSpecialField()))
  val c = IO(new BundleWithSpecialField())

  c.special := true.B // must initialize it

  c.exclude(_.special) :<>= p.exclude(_.special)
}
```

This generates the following Verilog, where the `special` field is not connected:

```scala mdoc:verilog
getVerilogString(new Example15)
```

## Techniques for connecting structurally inequivalent Chisel types

`DataView` and `viewAsSupertype` create a view of the component that has a different Chisel type.
This means that a user can first create a `DataView` of the consumer or producer (or both) so that the Chisel types are structurally equivalent.
This is useful when the difference between the consumer and producers aren't super nested, and also if they have rich Scala types which encode their structure.
In general, `DataView` is the preferred mechanism to use (if you can) because it maintains the most about of Chisel information in the Scala type, but there are many instances where it doesn't work and thus one must fall back on `Connectable`.

`Connectable` does not change the Chisel type, but instead changes the semantics of the operator to not error on the waived members if they are dangling or unconnected.
This is useful for when differences between the consumer and producer do not show up in the Scala type system (e.g. present/missing fields of type `Option[Data]`, or anonymous `Record`s) or are deeply nested in a bundle that is especially onerous to create a `DataView`.

Static casts (e.g. `(x: T)`) allows connecting components that have different Scala types, but leaves the Chisel type unchanged.
Use this to force a connection to occur, even if the Scala types are different.

> One may wonder why the operators require identical Scala types in the first place, if they can easily be bypassed.
The reason is to encourage users to use the Scala type system to encode Chisel information as it can make their code more robust; however, we don't want to be draconian about it because there are times when we want to enable the user to "just connect the darn thing".

When all else fails one can always manually expand the connection to do what they want to happen, member by member.
The down-side to this approach is its verbosity and that adding new members to a component will require updating the manual connections.

Things to remember about `Connectable` vs `viewAsSupertype`/`DataView` vs static cast (e.g. `(x: T)`):

- `DataView` and `viewAsSupertype` will preemptively remove members that are not present in the new view which has a different Chisel type, thus `DataView` *does* affect what is connected
- `Connectable` can be used to waive the error on members who end up being dangling or unconnected.
Importantly, `Connectable` waives *do not* affect what is connected
- Static cast does not remove extra members, thus a static cast *does not* affect what is connected

### Connecting different sub-types of the same super-type, with colliding names

In these examples, we are connecting `MyDecoupled` with `MyDecoupledOtherBits`.
Both are subtypes of `MyReadyValid`, and both have a `bits` field of `UInt(32.W)`.

The first example will use `.viewAsSupertype` to connect them as `MyReadyValid`.
Because it changes the Chisel type to omit both `bits` fields, the `bits` fields are unconnected.

```scala mdoc:silent
import experimental.dataview._
class MyDecoupledOtherBits extends MyReadyValid {
  val bits = UInt(32.W)
}
class Example12 extends RawModule {
  val in  = IO(Flipped(new MyDecoupled))
  val out = IO(new MyDecoupledOtherBits)

  out := DontCare

  out.viewAsSupertype(new MyReadyValid) :<>= in.viewAsSupertype(new MyReadyValid)
}
```

Note that the `bits` fields are unconnected.

```scala mdoc:verilog
getVerilogString(new Example12)
```

The second example will use a static cast and `.waive(_.bits)` to connect them as `MyReadyValid`.
Note that because the static cast does not change the Chisel type, the connection finds that both consumer and producer have a `bits` field.
This means that since they are structurally equivalent, they match and are connected.
The `waive(_.bits)` does nothing, because the `bits` are not dangling nor unconnected.



```scala mdoc:silent
import experimental.dataview._
class Example13 extends RawModule {
  val in  = IO(Flipped(new MyDecoupled))
  val out = IO(new MyDecoupledOtherBits)

  out := DontCare

  out.waiveAs[MyReadyValid](_.bits) :<>= in.waiveAs[MyReadyValid](_.bits)
}
```

Note that the `bits` fields ARE connected, even though they are waived, as `waive` just changes whether an error should be thrown if they are missing, NOT to not connect them if they are structurally equivalent. To always omit the connection, use `exclude` on one side and either `exclude` or `waive` on the other side.

```scala mdoc:verilog
getVerilogString(new Example13)
```

### Connecting sub-types to super-types by waiving extra members

> Note that in this example, it would be better to use `.viewAsSupertype`.

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
  val in  = IO(Flipped(new MyDecoupled))
  val out = IO(new MyReadyValid)
  out :<>= in.waiveAs[MyReadyValid](_.bits)
}
```

This generates the following Verilog, where `ready` and `valid` are connected, and `bits` is ignored:

```scala mdoc:verilog
getVerilogString(new Example5)
```

### Connecting different sub-types

> Note that in this example, it would be better to use `.viewAsSupertype`.

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
  val in  = IO(Flipped(new HasBits))
  val out = IO(new HasEcho)
  out.waiveAs[MyReadyValid](_.echo) :<>= in.waiveAs[MyReadyValid](_.bits)
}
```

This generates the following Verilog, where `ready` and `valid` are connected, and `bits` and `echo` are ignored:

```scala mdoc:verilog
getVerilogString(new Example7)
```

## FAQ

### How do I connect two items as flexibly as possible (try your best but never error)

Use `.unsafe` (both waives and allows squeezing of all fields).

```scala mdoc:silent
class ExampleUnsafe extends RawModule {
  val in  = IO(Flipped(new Bundle { val foo = Bool(); val bar = Bool() }))
  val out = IO(new Bundle { val baz = Bool(); val bar = Bool() })
  out.unsafe :<>= in.unsafe // bar is connected, and nothing errors
}
```

### How do I connect two items but don't care about the scala types being equivalent?

Use `.as` (upcasts the Scala type).

```scala mdoc:silent
class ExampleAs extends RawModule {
  val in  = IO(Flipped(new Bundle { val foo = Bool(); val bar = Bool() }))
  val out = IO(new Bundle { val foo = Bool(); val bar = Bool() })
  // foo and bar are connected, although Scala types aren't the same
  out.as[Data] :<>= in.as[Data]
}
```
