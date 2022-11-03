---

layout: docs

title:  "Connectable Operators"

section: "chisel3"

---

## Terminology

- "Chisel type" - a `Data` that is not bound to hardware
- "component" - a `Data` that is bound to hardware (`IO`, `Reg`, `Wire`, etc.)
- "member" - a child Chisel type/component of a parent Chisel type or component
- "field" - a named member of a `Record` or `Bundle` Chisel type or component

For more details about these concepts, please read (TODO link to Scala types vs Chisel types)

## Overview

The `Connectable` operators are the standard way to connect Chisel hardware components to one another.

Note: For descriptions of the semantics for the previous operators, see `connection-operators.md`.

All connection operators require the two hardware components (consumer and producer) to be Chisel type-equivalent (matching bundle field names and types (Record vs Vector vs Element), vector sizes, ground types (UInt/SInt/Bool/Clock etc)). Use `DataMirror.checkTypeEquivalence` to check this property.

The one exception to the type-equivalence rule is using the `WaivedData` mechanism, detailed at [section](#waived-data) at the end of this document.

Aggregate (`Record`, `Vec`, `Bundle`) Chisel types can include data members which are flipped relative to one another. Due to this, there are many desired connection behaviors between two Chisel components. The following are the Chisel connection operators:
 - `c := p` (mono-direction): assigns all p members to c; requires c & p to not have any flipped members
 - `c :#= p` (coercing mono-direction): assigns all p members to c; regardless of alignment
 - `c :<= p` (aligned-direction); assigns all aligned (non-flipped) c members from p
 - `c :>= p` (flipped-direction); assigns all flipped p members from c
 - `c :<>= p` (bi-direction operator); assigns all aligned c members from p; all flipped p members from c

## Alignment: Flipped vs Aligned

A member's alignment is a relative property; a member is aligned/flipped relative to another member. Hence, one must always say whether a member is flipped/aligned *with respect to (w.r.t)* another member of that type (parent, sibling, child etc.).

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

> Future work will refactor how these primitives are exposed to the user to make Chisel's type system more intuitive.

With this in mind, we can consider the following examples and detail relative alignments of members.

First, we can use a similar example to `Parent` but use `Input/Output` instead of `Flipped`.
Because `alignedChild` and `flippedChild` are non-aggregates, `Input` is basically just a `Flipped` and thus the alignments are unchanged compared to the previous `Parent` example.

```scala mdoc:silent
import chisel3._
class ParentWithOutputInput extends Bundle {
  val outputChild = Output(UInt(32.W)) // Equivalent to just UInt(32.W)
  val inputChild = Input(UInt(32.W))  // Equivalent to Flipped(UInt(32.W))
}
class MyModule2 extends Module {
  val p = Wire(new ParentWithOutputInput)
}
```

The aligments are the same as the previous `Parent` example:
 - `p` is aligned w.r.t `p`
 - `p.outputChild` is aligned w.r.t `p.outputChild`
 - `p.inputChild` is aligned w.r.t `p.inputChild`
 - `p` is aligned w.r.t `p.outputChild`
 - `p` is flipped w.r.t `p.inputChild`
 - `p.outputChild` is flipped w.r.t `p.inputChild`

The next example has a nested bundle `GrandParent` who instantiates an `Output` `ParentWithOutputInput` field and an `Input` `ParentWithOutputInput` field.

```scala mdoc:silent
import chisel3._
class GrandParent extends Bundle {
  val o = Output(new ParentWithOutputInput())
  val i = Input(new ParentWithOutputInput())
}
class MyModule1 extends Module {
  val g = Wire(new GrandParent)
}
```

Remember that `Output(gen)/Input(gen)` recursively strip the `Flipped` of any recursive children.
This makes every member of `gen` aligned with every other member of `gen`.

Consider the following alignments between grandparent and grandchildren. Because `o` and `i` have recursively stripped the flips of children, they are fully aligned. Thus, only their alignment to `g` influences grandchildren alignment:
 - `g` is aligned w.r.t `g.o.outputChild`
 - `g` is aligned w.r.t `g.o.inputChild`
 - `g` is flipped w.r.t `g.i.inputChild`
 - `g` is flipped w.r.t `g.i.outputChild`

Consider the following alignment relationships starting from `g.o` and `g.i`. *Note that whether `g.o` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.o` and `g.o.outputChild` because alignment is only relative to the two members in question! Because alignment is forced, everything is aligned between `g.o`/`g.i` and their children*:
 - `g.o` is aligned w.r.t. `g.o.outputChild`
 - `g.i` is aligned w.r.t. `g.i.outputChild`
 - `g.o` is aligned w.r.t. `g.o.inputChild`
 - `g.i` is aligned w.r.t. `g.i.inputChild`

In summary, `Input(gen)` and `Output(gen)` recursively coerce children alignment, as well as dictate `gen`'s alignment to its parent bundle (if it exists).

## Connecting components with fully aligned members

For simple connections where all members of aligned (non-flipped) with one another, use `:=`:


```scala mdoc:silent
import chisel3._
class FullyAlignedBundle extends Bundle {
  val a = Bool()
  val b = Bool()
}
class Example0 extends RawModule {
  val foo = IO(Flipped(new FullyAlignedBundle))
  val bar = IO(new FullyAlignedBundle)
  bar := foo
}
```

This generates the following Verilog, where each member of `foo` drives every member of `bar`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example0())
```

## Connections for components with members of mixed-alignment

Chisel types can include data members which are flipped relative to one another; in the example below, `a` and `b` are aligned/flipped relative to `MixedAlignmentBundle`.

```scala mdoc:silent
import chisel3._
class MixedAlignmentBundle extends Bundle {
  val a = Bool()
  val b = Flipped(Bool())
}
```

Due to this, there are many desired connection behaviors between two Chisel components. First we will investigate a common source of confusion between port-direction and connection-direction. Then, we will dive in to the Chisel connection operators useful for connecting components with members of mixed-alignments.

### Port-Direction Computation versus Connection-Direction Computation

A common question is if you use a mixed-alignment connection (such as `:<>=`) to connect submembers of parent components, does the alignment of the submember to their parent affect anything? The answer is no, because *alignment is always computed relative to what is being connected to, and members are always aligned with themselves.*

In the following example connecting `foo.a` to `bar.a`, whether `foo.a` is aligned with `foo` is irrelevant because the `:<>=` only computes alignment relative to the thing being connected to, and `foo.a` is aligned with `foo.a`.

```scala mdoc:silent
class Example1a extends RawModule {
  val foo = IO(Flipped(new MixedAlignmentBundle))
  val bar = IO(new MixedAlignmentBundle)
  bar.a :<>= foo.a // whether foo.a is aligned/flipped to foo is IRRELEVANT to what gets connected with :<>=
}
```

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example1a())
```

While `foo.b`'s alignment with `foo` does not affect our operators, it does influence whether `foo.b` is an output or input port of my module.
A common source of confusion is to mistake the process for determining whether `foo.b` is an output/input (the port-direction computation) with the process for determing how `:<>=` connects who to who (the connection-direction computation).
While both processes consider relative alignment, they are distinct.

The port-direction computation always computes alignment relative to the component marked with `IO`. An `IO(Flipped(gen))` is an input port, and any member of `gen` that is aligned/flipped with `gen` is an input/output port. An `IO(gen)` is an output port, and any member of `gen` that is aligned/flipped with `gen` is an output/input port.

The connection-direction computation always computes alignment based on explicit consumer/producer referenced for the connection. If one connects `foo :<>= bar`, alignments are computed based on `foo` and `bar`. If I connect `foo.a :<>= bar.a`, then alignments are computed based on `foo.a` and `bar.a` (and the alignment of `foo` to `foo.a` is irrelevant).

This means that users can try to assign to input ports of their module! If I write `x :<>= y`, and `x` is an input to my module, then that is what the connection is trying to do. However, because input ports are not assignable from within my module, Chisel will throw an error. This is the same error a user would get using a mono-directioned operator: `x := y` will throw the same error if `x` is an input module. *Whether a component is assignable is irrelevant to the semantics of any connection operator assigning to it.*

In summary, the port-direction computation is relative to the root marked `IO`, but connection-direction computation is relative to the consumer/producer that the connection is doing. This has the positive property that connection semantics are solely based on the Chisel types of the consumer/producer (nothing more, nothing less).

### Bi-direction connection operator (:<>=)

For connections where you want 'bulk-connect-like-semantics' where the aligned members are driven producer-to-consumer and flipped members are driven consumer-to-producer, use `:<>=`.

```scala mdoc:silent
class Example1 extends RawModule {
  val foo = IO(Flipped(new MixedAlignmentBundle))
  val bar = IO(new MixedAlignmentBundle)
  bar :<>= foo
}
```

This generates the following Verilog, where the aligned members are driven `foo` to `bar` and flipped members are driven `bar` to `foo`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example1())
```

### Aligned connection operator (:<=)

For connections where you want the aligned-half of 'bulk-connect-like-semantics' where the aligned members are driven producer-to-consumer and flipped members are ignored, use `:<=` (the "aligned connection").

```scala mdoc:silent
class Example2 extends RawModule {
  val foo = IO(Flipped(new MixedAlignmentBundle))
  val bar = IO(new MixedAlignmentBundle)
  foo.b := DontCare // Otherwise FIRRTL throws an uninitialization error
  bar :<= foo
}
```

This generates the following Verilog, where the aligned members are driven `foo` to `bar` and flipped members are ignored:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example2())
```

### Flipped connection operator (:>=)

For connections where you want the flipped-half of 'bulk-connect-like-semantics' where the aligned members are ignored and flipped members are assigned consumer-to-producer, use `:<=` (the "flipped connection", or "backpressure connection").

```scala mdoc:silent
class Example3 extends RawModule {
  val foo = IO(Flipped(new MixedAlignmentBundle))
  val bar = IO(new MixedAlignmentBundle)
  bar.a := DontCare // Otherwise FIRRTL throws an uninitialization error
  bar :>= foo
}
```

This generates the following Verilog, where the aligned members are ignore and the flipped members are driven `bar` to `foo`:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example3())
```

### Coercing mono-direction connection operator (:#=)

For connections where you want to every producer member to always drive every consumer member, regardless of alignment, use `:#=` (the "coercion connection"). This operator is useful for initializing wires whose types contain members of mixed alignment.

```scala mdoc:silent
import chisel3.experimental.BundleLiterals._
class Example4 extends RawModule {
  val bar = Wire(new MixedAlignmentBundle)
  dontTouch(bar) // So we see it in the output verilog
  bar :#= (new MixedAlignmentBundle).Lit(_.a -> true.B, _.b -> true.B)
}
```

This generates the following Verilog, where all members are driven `foo` to `bar`, regardless of alignment:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example4())
```

## Waived Data

It is not uncommon for a user to want to connect Chisel components which are not type equivalent. For example, a user may want to cook up the `ready`/`valid` members of a `ReadyValidIO` to a `DecoupledIO`, but
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
import scala.collection.immutable.SeqMap
import chisel3.experimental.AutoCloneType
class Example8 extends RawModule {
  val abType = new Record with AutoCloneType { def elements = SeqMap("a" -> Bool(), "b" -> Flipped(Bool())) }
  val bcType = new Record with AutoCloneType { def elements = SeqMap("b" -> Flipped(Bool()), "c" -> Bool()) }

  val p = Wire(abType)
  val c = Wire(bcType)

  //p :#= abType.Lit(_.elements("a") -> true.B, _.elements("b") -> true.B)
  //c :#= bcType.Lit(_.elements("b") -> true.B, _.elements("c") -> true.B)

  //c.waiveAll :<>= p.waiveAll
}
```

This generates the following Verilog, where the `b` member is driven from `c` to `p`, and `a` and `c` members are initialized to default values:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example8())
```


### Always ignore extra members (partial connection operator)