---
layout: docs
title:  "Layers"
section: "chisel3"
---

# Layers

Layers are used to describe functionality of a Chisel circuit or module that a
user would like to _optionally_ include at Verilog elaboration time.  This
feature is intended to be used to optionally include verification or debug logic
that a user does not want to always have present.

Each layer is broken into two pieces:

1. A layer _declaration_
1. One or more _layer blocks_ inside modules in the circuit

There are two kinds of layers, each kind is lowered to verilog under a different
_convention_. The kinds of layers are:

1. "Extract" Layers: layers whose blocks are lowered to bound-in modules, and
2. "Inline" Layers: layers whose blocks are lowered to ifdefs macros.

To declare a layer, create a singleton `object` in scala that extends the

extend the abstract class `chisel3.layer.Layer`, passing into the layer
constructor either `chisel3.layer.LayerConfig.Extract()` for an "extract" layer,
or `chisel3.layer.LayerConfig.Inline` for "inline" layers.

Layers may be nested. To declare a nested layer, extend the
`chisel3.layer.Layer` abstract class inside another declaration.

The following example declares four layers:

```scala mdoc:silent
import chisel3.layer.{Layer, LayerConfig}

object A extends Layer(LayerConfig.Extract()) {
  object B extends Layer(LayerConfig.Extract()) {
    object C extends Layer(LayerConfig.Extract())
  }
  object D extends Layer(LayerConfig.Extract())
}
```

A _layer block_, associated with a layer, adds optional functionality to a
module that is enabled if that layer is enabled.  Each layer block must refer to
a pre-declared layer.  Layer block nesting must match the nesting of declared
layers.

To define a layer block, use the `chisel3.layer.block` inside a Chisel module.
An layer block may use any Chisel or Scala value visible to its Scala lexical
scope.

The following example defines layer blocks inside module `Foo` and declares
wires which are connected to values captured from visible lexical scope:

```scala mdoc:silent
import chisel3._
import chisel3.layer.block

class Foo extends RawModule {
  val port = IO(Input(Bool()))

  block(A) {
    val a = WireInit(port)
    block(A.B) {
      val b = WireInit(a)
      block(A.B.C) {
        val c = WireInit(b)
      }
    }
    block(A.D) {
      val d = WireInit(port ^ a)
    }
  }
}
```

## Conventions

Currently, there is only one supported convention, `Bind`.  This will cause layer
blocks to be lowered to Verilog modules that are instantiated via the
SystemVerilog `bind` mechanism.  The lowering to Verilog of layer blocks avoids
illegal nested usage of `bind`.

More conventions may be supported in the future.

## Built-in Layers and User-defined Layers

Chisel provides several built-in layers.  These are shown below with their full
Scala paths:

```
`chisel3.layers.Verification`
├── `chisel3.layers.Verification.Assert`
├── `chisel3.layers.Verification.Assume`
└── `chisel3.layers.Verification.Cover`
```

These built-in layers are dual purpose.  First, they are layers that match
common use cases for sequestering verification code.  The `Verification` layer
is for common verification collateral.  The `Assert`, `Assume`, and `Cover`
layers are for, respectively, assertions, assumptions, and cover statements.
Second, the Chisel standard library uses them for some of its APIs.

For predictability of output, these layers will always be show up in the FIRRTL
that Chisel emits.

A user is free to define their own layers, as shown previously with layer `A`,
`A.B`, etc.  User-defined layers are only emitted into FIRRTL if they have layer
block users.  Layers can be unconditionally emitted using the
`chisel3.layer.addLayer` API.

## Examples

### Design Verification Example

Consider a use case where a design or design verification engineer would like to
add some asserts and debug prints to a module.  The logic necessary for the
asserts and debug prints requires additional computation.  All of this code
should selectively included at Verilog elaboration time (not at Chisel
elaboration time).  The engineer can use three layers to do this.

There are three layers used in this example:

1. The built-in `Verification` layer
1. The built-in `Assert` layer which is nested under the built-in `Verification`
   layer
1. A user-defined `Debug` layer which is also nested under the built-in
   `Verification` layer

The `Verification` layer can be used to store common logic used by both the
`Assert` and `Debug` layers.  The latter two layers allow for separation of,
respectively, assertions from prints.

One way to write this in Scala is the following:

```scala mdoc:reset:silent
import chisel3._
import chisel3.layer.{Layer, LayerConfig, block}
import chisel3.layers.Verification

// User-defined layers are declared here.  Built-in layers do not need to be declared.
object UserDefined {
  // Define an implicit val `root` of type `Layer` to cause layers which can see
  // this to use `root` as their parent layer.  This allows us to nest the
  // user-defined `Debug` layer under the built-in `Verification` layer.
  implicit val root = Verification
  object Debug extends Layer(LayerConfig.Extract())
}

class Foo extends Module {
  val a = IO(Input(UInt(32.W)))
  val b = IO(Output(UInt(32.W)))

  b := a +% 1.U

  // This adds a `Verification` layer block inside Foo.
  block(Verification) {

    // Some common logic added here.  The input port `a` is "captured" and
    // used here.
    val a_d0 = RegNext(a)

    // This adds an `Assert` layer block.
    block(Verification.Assert) {
      chisel3.assert(a >= a_d0, "a must always increment")
    }

    // This adds a `Debug` layer block.
    block(UserDefined.Debug) {
      printf("a: %x, a_d0: %x", a, a_d0)
    }

  }

}

```

After compilation, this will produce three layer include files with the
following filenames.  One file is created for each layer:

1. `layers_Foo_Verification.sv`
1. `layers_Foo_Verification_Assert.sv`
1. `layers_Foo_Verification_Debug.sv`

A user can then include any combination of these files in their design to
include the optional functionality describe by the `Verification`, `Assert`, or
`Debug` layers.  The `Assert` and `Debug` bind files automatically include the
`Verification` bind file for the user.

#### Implementation Notes

_Note: the names of the modules and the names of any files that contain these
modules are FIRRTL compiler implementation defined!  The only guarantee is the
existence of the three layer include files.  The information in this subsection
is for informational purposes to aid understanding._

In implementation, a FIRRTL compiler creates four Verilog modules for the
circuit above (one for `Foo` and one for each layer block in module `Foo`):

1. `Foo`
1. `Foo_Verification`
1. `Foo_Verification_Assert`
1. `Foo_Verification_Cover`
1. `Foo_Verification_Debug`

These will typically be created in separate files with names that match the
modules, i.e., `Foo.sv`, `Foo_Verification.sv`, `Foo_Verification_Assert.sv`,
`Foo_Verification_Debug.sv`.

The ports of each module created from a layer block will be automatically
determined based on what that layer block captured from outside the layer block.
In the example above, the `Verification` layer block captured port `a`.  Both
the `Assert` and `Debug` layer blocks captured `a` and `a_d0`.  Layer blocks may
be optimized to remove/add ports or to move logic into a layer block.

#### Verilog Output

The complete Verilog output for this example is reproduced below:

```scala mdoc:verilog
// Use ChiselStage instead of chisel3.docs.emitSystemVerilog because we want layers printed here (obviously)
import circt.stage.ChiselStage
ChiselStage.emitSystemVerilog(new Foo, firtoolOpts=Array("-strip-debug-info", "-disable-all-randomization"))
```
