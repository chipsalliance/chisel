---
layout: docs
title:  "Bundles and Vecs"
section: "chisel3"
---

# Bundles and Vecs

`Bundle` and `Vec` are classes that allow the user to expand the set of Chisel datatypes with aggregates of other types.

Bundles group together several named fields of potentially different types into a coherent unit, much like a `struct` in
C. Users define their own bundles by defining a class as a subclass of `Bundle`.

```scala mdoc:silent
import chisel3._
class MyFloat extends Bundle {
  val sign        = Bool()
  val exponent    = UInt(8.W)
  val significand = UInt(23.W)
}

class ModuleWithFloatWire extends RawModule {
  val x  = Wire(new MyFloat)
  val xs = x.sign
}
```

You can create literal Bundles using the experimental [Bundle Literals](../appendix/experimental-features#bundle-literals) feature.

Scala convention is to name classes using UpperCamelCase, and we suggest you follow that convention in your Chisel code.

Vecs create an indexable vector of elements, and are constructed as follows:

```scala mdoc:silent
class ModuleWithVec extends RawModule {
  // Vector of 5 23-bit signed integers.
  val myVec = Wire(Vec(5, SInt(23.W)))

  // Connect to one element of vector.
  val reg3 = myVec(3)
}
```

(Note that we specify the number followed by the type of the `Vec` elements. We also specifiy the width of the `SInt`)

The set of primitive classes
(`SInt`, `UInt`, and `Bool`) plus the aggregate
classes (`Bundles` and `Vec`s) all inherit from a common
superclass, `Data`.  Every object that ultimately inherits from
`Data` can be represented as a bit vector in a hardware design.

Bundles and Vecs can be arbitrarily nested to build complex data
structures:

```scala mdoc:silent
class BigBundle extends Bundle {
 // Vector of 5 23-bit signed integers.
 val myVec = Vec(5, SInt(23.W))
 val flag  = Bool()
 // Previously defined bundle.
 val f     = new MyFloat
}
```

Note that the builtin Chisel primitive and aggregate classes do not
require the `new` when creating an instance, whereas new user
datatypes will.  A Scala `apply` constructor can be defined so
that a user datatype also does not require `new`, as described in
[Function Constructor](../explanations/functional-module-creation).

### Flipping Bundles

The `Flipped()` function recursively flips all elements in a Bundle/Record. This is very useful for building bidirectional interfaces that connect to each other (e.g. `Decoupled`). See below for an example.

```scala mdoc:silent
class ABBundle extends Bundle {
  val a = Input(Bool())
  val b = Output(Bool())
}
class MyFlippedModule extends RawModule {
  // Normal instantiation of the bundle
  // 'a' is an Input and 'b' is an Output
  val normalBundle = IO(new ABBundle)
  normalBundle.b := normalBundle.a

  // Flipped recursively flips the direction of all Bundle fields
  // Now 'a' is an Output and 'b' is an Input
  val flippedBundle = IO(Flipped(new ABBundle))
  flippedBundle.a := flippedBundle.b
}
```

This generates the following Verilog:

```scala mdoc:verilog
import circt.stage.ChiselStage

ChiselStage.emitSystemVerilog(new MyFlippedModule())
```

### MixedVec

(Chisel 3.2+)

All elements of a `Vec` must have the same parameterization. If we want to create a Vec where the elements have the same type but different parameterizations, we can use a MixedVec:

```scala mdoc:silent
import chisel3.util.MixedVec
class ModuleMixedVec extends Module {
  val io = IO(new Bundle {
    val x = Input(UInt(3.W))
    val y = Input(UInt(10.W))
    val vec = Output(MixedVec(UInt(3.W), UInt(10.W)))
  })
  io.vec(0) := io.x
  io.vec(1) := io.y
}
```

We can also programmatically create the types in a MixedVec:

```scala mdoc:silent
class ModuleProgrammaticMixedVec(x: Int, y: Int) extends Module {
  val io = IO(new Bundle {
    val vec = Input(MixedVec((x to y) map { i => UInt(i.W) }))
    // ...
  })
  // ...rest of the module goes here...
}
```

### A note on `cloneType` (For Chisel < 3.5)

NOTE: This section **only applies to Chisel before Chisel 3.5**.
As of Chisel 3.5, `Bundle`s should **not** `override def cloneType`,
as this is a compiler error when using the chisel3 compiler plugin for inferring `cloneType`.

Since Chisel is built on top of Scala and the JVM,
it needs to know how to construct copies of `Bundle`s for various
purposes (creating wires, IOs, etc).
If you have a parametrized `Bundle` and Chisel can't automatically figure out how to
clone it, you will need to create a custom `cloneType` method in your bundle.
In the vast majority of cases, **this is not required**
as Chisel can figure out how to clone most `Bundle`s automatically:

```scala mdoc:silent
class MyCloneTypeBundle(val bitwidth: Int) extends Bundle {
   val field = UInt(bitwidth.W)
   // ...
}
```

The only caveat is if you are passing something of type `Data` as a "generator" parameter,
in which case you should make it a `private val`, and define a `cloneType` method with
`override def cloneType = (new YourBundleHere(...)).asInstanceOf[this.type]`.

For example, consider the following `Bundle`. Because its `gen` variable is not a `private val`, the user has to
explicitly define the `cloneType` method:

<!-- Cannot compile this because the cloneType is now an error -->
```scala
import chisel3.util.{Decoupled, Irrevocable}
class RegisterWriteIOExplicitCloneType[T <: Data](gen: T) extends Bundle {
  val request  = Flipped(Decoupled(gen))
  val response = Irrevocable(Bool())
  override def cloneType = new RegisterWriteIOExplicitCloneType(gen).asInstanceOf[this.type]
}
```

We can make this this infer cloneType by making `gen` private since it is a "type parameter":

```scala mdoc:silent
import chisel3.util.{Decoupled, Irrevocable}
class RegisterWriteIO[T <: Data](private val gen: T) extends Bundle {
  val request  = Flipped(Decoupled(gen))
  val response = Irrevocable(Bool())
}
```
