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
