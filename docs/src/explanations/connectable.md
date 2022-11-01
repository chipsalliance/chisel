---

layout: docs

title:  "Connectable Operators"

section: "chisel3"

---

The `Connectable` operators are the standard way to connect Chisel hardware components to one another.

Note: For descriptions of the semantics for the previous operators, see `connection-operators.md`

All connection operators require the two hardware components (consumer and producer) to be Chisel type-equivalent (matching bundle field names and types, vector sizes, ground types (UInt/SInt)). Use `DataMirror.checkTypeEquivalence` to check this property.

The one exception to the type-equivalence rule is using the `WaivedData` mechansim, detailed at the end of this document.

Chisel types can include data members which are flipped relative to one another. Due to this, there are many desired connection behaviors between two Chisel components. The following are the Chisel connection operators:
 - `c := p` (mono-direction): assigns all p fields to c; requires c & p to not have flips
 - `c :#= p` (coercing mono-direction): assigns all p fields to c; ignores all flips
 - `c :<= p` (aligned-direction); assigns all aligned (non-flipped) c fields from p
 - `c :>= p` (flipped-direction); assigns all flipped p fields from c
 - `c :<>= p` (bi-direction operator); assigns all aligned c fields from p; all flipped p fields from c

## Alignment: Flipped vs Aligned

A field's alignment is a relative property; a field is aligned/flipped relative to another member. Hence, one must always say whether a field is flipped/aligned *with respect to (w.r.t)* another member of that type (parent, sibling, child etc.).

We use the following example of a non-nested bundle `Parent` to let us state all of the alignment relationships between members of `p`.

```scala mdoc:silent
import chisel3._
class Parent extends Bundle {
  val foo = UInt(32.W)
  val bar = Flipped(UInt(32.W))
}
class MyModule0 extends Module {
  val p = Wire(new Parent)
}
```

First, every member is always aligned with themselves:
 - `p` is aligned w.r.t `p`
 - `p.foo` is aligned w.r.t `p.foo`
 - `p.bar` is aligned w.r.t `p.bar`

Next, we list all parent/child relationships. Because the `bar` field is `Flipped`, it changes its aligment relative to its parent. 
 - `p` is aligned w.r.t `p.foo`
 - `p` is flipped w.r.t `p.bar`

Finally, we can list all sibling relationships:
 - `p.foo` is flipped w.r.t `p.bar`

The next example has a nested bundle `GrandParent` who instantiates an aligned `Parent` field and flipped `Parent` field.

```scala mdoc:silent
import chisel3._
class GrandParent extends Bundle {
  val parent = new Parent()
  val flippedParent = Flipped(new Parent())
}
class MyModule1 extends Module {
  val g = Wire(new GrandParent)
}
```

Consider the following alignements between grandparent and grandchildren. An odd number of flips indicate a flipped relationship; even numbers of flips indicate an aligned relationship.
 - `g` is aligned w.r.t `g.flippedParent.bar`
 - `g` is aligned w.r.t `g.parent.foo`
 - `g` is flipped w.r.t `g.flippedParent.foo`
 - `g` is flipped w.r.t `g.parent.bar`

Consider the following alignment relationships starting from `g.parent` and `g.flippedParent`. *Note that whether `g.parent` is aligned/flipped relative to `g` has no effect on the aligned/flipped relationship between `g.parent` and `g.parent.foo` because alignment is only relative to the two members in question!*:
 - `g.parent` is aligned w.r.t. `g.parent.foo`
 - `g.flippedParent` is aligned w.r.t. `g.flippedParent.foo`
 - `g.parent` is flipped w.r.t. `g.parent.bar`
 - `g.flippedParent` is flipped w.r.t. `g.flippedParent.bar`

In summary, a field is aligned or flipped w.r.t. another member of the hardware component. This means that the type of the consumer/producer is the only information needed to determine the behavior of any operator. *Whether the consumer/producer is a subfield of a larger bundle is irrelevant; you ONLY need to know the type of the consumer/producer*.

## Input/Output/IO (Maybe talk about later once I fix Chisel directionality?)

`Input(gen)`/`Output(gen)` are coercing operators. They perform two functions: (1) create a new Chisel type that has all flips removed from all recursive children fields but structurally equivalent to `gen`, and (2) apply `Flipped` if `Input`, keep aligned (do nothing) if `Output`.

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

Due to this, there are many desired connection behaviors between two Chisel components. The following are the Chisel connection operators useful for connecting components with members of mixed-alignments.

### Bi-direction connection operator (:<>=)

For connections where you want 'bulk-connect-like-semantics' where the aligned fields are driven producer-to-consumer and flipped fields are driven consumer-to-producer, use `:<>=`.

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

For connections where you want the aligned-half of 'bulk-connect-like-semantics' where the aligned fields are driven producer-to-consumer and flipped fields are ignored, use `:<=` (the "aligned connection").

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

For connections where you want the flipped-half of 'bulk-connect-like-semantics' where the aligned fields are ignored and flipped fields are assigned consumer-to-producer, use `:<=` (the "flipped connection", or "backpressure connection").

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

For connections where you want to every producer field to always drive every consumer field, regardless of alignment, use `:#=` (the "coercion connection"). This operator is useful for initializing wires whose types contain members of mixed alignment.

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

It is not uncommon for a user to want to connect Chisel components which are not type equivalent. For example, a user may want to cook up the `ready`/`valid` fields of a `ReadyValidIO` to a `DecoupledIO`, but
because the `bits` field is not present in both, our operators would reject a connection.

`WaivedData` is the mechanism to specialize connection operator behavior in these scenarios. For any addition field which is not present in the other component being connected to, they can be explicitly waived from the operator to be ignored, rather than trigger an error.

### Connecting sub-types to super-types by waiving extra fields

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

### Connecting types with optional fields

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


A not uncommon usecase is to try to connect two Records; for matching fields, they should be connected, but for unmatched fields, they should be assigned a default value. To accomplish this, use the other operators to initialize all Record fields, then use `:<>=` with `waiveAll` to connect only the matching fields.


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

This generates the following Verilog, where the `b` field is driven from `c` to `p`, and `a` and `c` fields are initialized to default values:

```scala mdoc:verilog
import chisel3.stage.ChiselStage

ChiselStage.emitVerilog(new Example8())
```


### Always ignore extra fields (partial connection operator)