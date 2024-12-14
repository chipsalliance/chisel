---
layout: docs
title:  "Layers"
section: "chisel3"
---

# Layers

Layers describe functionality of a Chisel generator that a user would like to
_optionally_ include at Verilog elaboration time.  Pragmatically, they are a
feature to access SystemVerilog's `bind` construct and `` `ifdef `` preprocessor
macros.  Layers are typically used to describe design verification code or
debugging logic that a user would like to be able to later disable (for
performance, verbosity, or cleanliness reasons) or use internally, but exclude
from delivery to a customer.

## Overview

A layer consists of two pieces:

1. A layer _declaration_, and
1. One or more _layer blocks_ inside Chisel modules.

The declaration indicates that optional functionality can exist.  The layer
block contains the optional functionality.

There are two kinds of layers.  The layer kind determines the _convention_,
i.e., how the layer blocks of a layer are represented in Verilog and the
mechanism to enable a layer.  Available layer kinds are:

1. "Extract" Layers: layers whose blocks are lowered to modules that are
   instantiated using `bind` and can be enabled by including a file during
   Verilog elaboration, and
2. "Inline" Layers: layers whose blocks will be guarded with `` `ifdef `` macros
   and can be enabled by setting a Verilog preprocessor define.

Extract layers may also specify a directory into which their collateral are
written.

:::info

For more information about these SystemVerilog concepts, the IEEE 1800-2023
standard discusses `bind` in Section 23.11 and `` `ifdef `` in Section 23.6.

:::

To declare a layer, create a singleton `object` in scala that extends the
abstract class `chisel3.layer.Layer`, passing into the layer constructor either
an object of class `chisel3.layer.LayerConfig.Extract` for an extract layer, or
the object `chisel3.layer.LayerConfig.Inline` for an inline layer.

Below, an extract layer and an inline layer are declared:

```scala mdoc:silent
import chisel3.layer.{Layer, LayerConfig}

object A extends Layer(LayerConfig.Extract())

object B extends Layer(LayerConfig.Inline)
```

Layers may be nested.  Nesting a child layer under a parent layer means that the
child layer may access constructs in the parent layer.  Put differently, the
child layer will only be enabled if the parent layer is already enabled.  To
declare a nested layer, extend the `chisel3.layer.Layer` abstract class inside
another declaration.

The following example defines an extract layer with two nested layers inside it:

```scala mdoc:silent
object C extends Layer(LayerConfig.Extract()) {
  object D extends Layer(LayerConfig.Extract())
  object E extends Layer(LayerConfig.Inline) {
    object F extends Layer(LayerConfig.Inline)
  }
}
```

:::info

SystemVerilog prohibits a `bind` instantiation under another `bind`
instantiation.  However, Chisel allows nesting of extract layers.  This is
resolved by the FIRRTL compiler to restructure nested extract layers to be
sibling modules that communicate via ports.

:::

:::warning

Extract layers may not be nested under inline layers.  However, inline layers
may be nested under extract layers.

Any module which contains layer blocks or transitively contains layer blocks in
its submodules may not be instantiated under a layer block.

:::

A _layer block_, associated with a layer, adds optional functionality to a
module that is enabled if that layer is enabled.  To define a layer block, use
the `chisel3.layer.block` inside a Chisel module and pass the layer that it
should be associated with.

Inside the layer block, any Chisel or Scala value visible in lexical scope may
be used.  Layer blocks may not return values.  Any values created inside a layer
block are not accessible outside the layer block, unless using layer-colored
probes.

The following example defines layer blocks inside module `Foo`.  Each layer
block contains a wire that captures a value from its visible lexical scope.
(For nested layer blocks, this scope includes their parent layer blocks.):

```scala mdoc:silent
import chisel3._
import chisel3.layer.block

class Foo extends RawModule {
  val port = IO(Input(Bool()))

  block(A) {
    val a = WireInit(port)
  }

  block(B) {
    val b = WireInit(port)
  }

  block(C) {
    val c = WireInit(port)
    block(C.D) {
      val d = WireInit(port | c)
    }
    block(C.E) {
      val e = WireInit(port ^ c)
      block (C.E.F) {
        val f = WireInit(port & e)
      }
    }
  }
}
```

The layer block API will automatically create parent layer blocks for you if
possible.  In the following example, it is legal to directly create a layer
block of `C.D` directly in a module:

```scala mdoc:silent
class Bar extends RawModule {
  block (C.D) {}
}
```

Formally, it is legal to create a layer block associated with a layer as long as
the current scope is an _ancestor_ of the request layer.

:::info

The requirement is an _ancestor_ relationship, not a _proper ancestor_
relationship.  This means that it is legal to nest a layer block under a layer
block of the same layer like:

```scala mdoc:silent
class Baz extends RawModule {
  block(A) {
    block(A) {}
  }
}
```

:::

## Verilog ABI

Layers are compiled to SystemVerilog using the FIRRTL ABI.  This ABI defines
what happens to layer blocks in a Chisel design and how a layer can be enabled
after a design is compiled to SystemVerilog.

:::info

For the exact definition of the FIRRTL ABI for layers, see the [latest version
of the FIRRTL ABI
Specification](https://github.com/chipsalliance/firrtl-spec/releases/latest/download/abi.pdf).

:::

### Extract Layers

Extract layers have their layer blocks removed from the design.  To enable a
layer, a file with a specific name should be included in the design.  This file
begins with `layers-` and then includes the circuit name and all layer names
delimited with dashes (`-`).

For example, for module `Foo` declared above, this will produce three files, one
for each extract layer:

```
layers-Foo-A.sv
layers-Foo-C.sv
layers-Foo-C-D.sv
```

To enable any of these layers at compilation time, the appropriate file should
be included in the build.  Any combination of files may be included.  Including
only a child layer's file will automatically include its parent layer's file.

### Inline Layers

Inline layers have their layer blocks guarded with conditional compilation
directives.  To enable an inline layer, set a preprocessor define when compiling
your design.  The preprocessor define begins with `layer_` and then includes the
circuit name and all layer names delimited with dollar signs (`$`).  Parent
extract layer names appear in the macro.

For example, for module `Foo` declared above, this will be sensitive to three
macros, one for each inline layer:

```
layer_Foo$B
layer_Foo$C$E
layer_Foo$C$E$F
```

## User-defined Layers

A user is free to define as many layers as they want.  All layers shown
previously are user-defined, e.g., `A` and `C.E` are user-defined layers.
User-defined layers are only emitted into FIRRTL if they have layer block users.
To change this behavior and unconditionally emit a user-defined layer, use the
`chisel3.layer.addLayer` API.

:::tip

Before creating new user-defined layers, consider using the built-in layers
defined below.  Additionally, if working in a larger project, the project may
have it's own user-defined layers that you are expected to use.  This is because
the ABIs affect the build system.  Please consult with a technical lead of the
project to see if this is the case.

:::

## Built-in Layers

Chisel provides several built-in layers.  These are shown below with their full
Scala paths.  All built-in layers are extract layers:

```
chisel3.layers.Verification
├── chisel3.layers.Verification.Assert
├── chisel3.layers.Verification.Assume
└── chisel3.layers.Verification.Cover
```

These built-in layers are dual purpose.  First, these layers match the common
use case of sequestering verification code.  The `Verification` layer is for
common verification collateral.  The `Assert`, `Assume`, and `Cover` layers are
for, respectively, assertions, assumptions, and cover statements.  Second, the
Chisel standard library uses them for a number of its APIs.  _Unless otherwise
wrapped in a different layer block, the following operations are automatically
placed in layers_:

* Prints are placed in the `Verification` layer
* Assertions are placed in the `Verification.Assert` layer
* Assumptions are placed in the `Verification.Assume` layer
* Covers are placed in the `Verification.Cover` layer

For predictability of output, these layers will always be show up in the FIRRTL
that Chisel emits.  To change this behavior, use `firtool` command line options
to _specialize_ these layers (remove their optionality by making them always
enabled or disabled).  Use `-enable-layers` to enable a layer, `-disable-layers`
to disable a layer, or `-default-layer-specialization` to set a default
specialization.

:::tip

Users may extend built-in layers with user-defined layers using an advanced API.
To do this, the layer parent must be specified as an implicit value.

The following example nests the layer `Debug` to the `Verification` layer:

```scala mdoc:silent
object UserDefined {
  // Define an implicit val `root` of type `Layer` to cause layers which can see
  // this to use `root` as their parent layer.  This allows us to nest the
  // user-defined `Debug` layer under the built-in `Verification` layer.
  implicit val root: Layer = chisel3.layers.Verification
  object Debug extends Layer(LayerConfig.Inline)
}
```

:::

## Examples

### Simple Extract Layer

The design below has a single extract layer that, when enabled, will add an
assert that checks for overflow.  Based on the FIRRTL ABI, we can expect that a
file called `layers-Foo-A.sv` will be produced when we compile it.

```scala mdoc:reset:silent
import chisel3._
import chisel3.layer.{Layer, LayerConfig, block}
import chisel3.ltl.AssertProperty

object A extends Layer(LayerConfig.Extract())

class Foo extends Module {
  val a, b = IO(Input(UInt(4.W)))
  val sum = IO(Output(UInt(4.W)))

  sum :<= a +% b

  block(A) {
    withDisable(Disable.Never) {
      AssertProperty(!(a +& b)(4), "overflow occurred")
    }
  }

}
```

After compilation, we get the following SystemVerilog.  Comments that include
`FILE` indicate the beginning of a new file:

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification",
    "-enable-layers=Verification.Assert",
    "-enable-layers=Verification.Assume",
    "-enable-layers=Verification.Cover"
  )
)
```

:::info

The above example was compiled with the firtool options
`-enable-layers=Verification`, `-enable-layers=Verification.Assert`,
`-enable-layers=Verification.Assume`, and `-enable-layers=Verification.Cover` to
make the output terser.  Normally, bind files would show up for these built-in
layers.

:::

:::info

Note: the generated module, `Foo_A`, and its file, `Foo_A.sv`, are _not part of
the ABI_.  You should not rely on any generated module names or files other than
the bind file, `layers-Foo-A.sv`.

:::

### Simple Inline Layer

The design below is the same as the previous example, but uses an inline layer.
Based on the FIRRTL ABI, we can expect that the body of the layer block will be
guarded by an `` `ifdef `` sensitive to the preprocessor macro `layer_Foo$A`.

```scala mdoc:reset:silent
import chisel3._
import chisel3.layer.{Layer, LayerConfig, block}
import chisel3.ltl.AssertProperty

object A extends Layer(LayerConfig.Inline)

class Foo extends Module {
  val a, b = IO(Input(UInt(4.W)))
  val sum = IO(Output(UInt(4.W)))

  sum :<= a +% b

  block(A) {
    withDisable(Disable.Never) {
      AssertProperty(!(a +& b)(4), "overflow occurred")
    }
  }

}
```

After compilation, we get the following SystemVerilog.

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(
  new Foo,
  firtoolOpts = Array(
    "-strip-debug-info",
    "-disable-all-randomization",
    "-enable-layers=Verification,Verification.Assert,Verification.Assume,Verification.Cover"
  )
)
```

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
  implicit val root: Layer = Verification
  object Debug extends Layer(LayerConfig.Inline)
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

After compilation, this will produce two layer include files with the
following filenames.  One file is created for each extract layer:

1. `layers_Foo_Verification.sv`
1. `layers_Foo_Verification_Assert.sv`

Additionally, the resulting SystemVerilog will be sensitive to the preprocessor
define `layer_Foo$Verification$Debug` due to the one inline layer we added.

A user can then include any combination of these files in their design to
include the optional functionality described by the `Verification` or
`Verification.Assert` layers and enable debugging by setting the preprocessor
macro.  The `Verification.Assert` bind file automatically includes the
`Verification` bind file for the user.

#### Implementation Notes

:::warning

This section describes the implementation of how layers are compiled.  Anything
that is _not_ a bind file name or a preprocessor macro should not be relied
upon!  A FIRRTL compiler may implement this differently or may optimize layer
blocks in any legal way it chooses.  E.g., layer blocks associated with the same
layer may be merged, layer blocks may be moved up or down the hierarchy, code
that only fans out to a layer block may be sunk into it, and unused layer blocks
may be deleted.

The information below is for user understanding and interest only.

:::

In implementation, a FIRRTL compiler creates three Verilog modules for the
circuit above (one for `Foo` and one for each layer block associated with an
extract layer in module `Foo`):

1. `Foo`
1. `Foo_Verification`
1. `Foo_Verification_Assert`

These will typically be created in separate files with names that match the
modules, i.e., `Foo.sv`, `Foo_Verification.sv`, and
`Foo_Verification_Assert.sv`.

The ports of each module created from a layer block will be automatically
determined based on what that layer block captured from outside the layer block.
In the example above, the `Verification` layer block captured port `a`.  The
`Assert` layer block captured captured `a` and `a_d0`.

:::info

Even though there are no layer blocks that use the `Verification.Assume` or
`Verification.Cover` layers, bind files which have no effect are produced in the
output.  This is due to the ABI which requires that layers that are defined in
FIRRTL must produce these files.

:::

#### Verilog Output

The complete Verilog output for this example is reproduced below:

```scala mdoc:verilog
// Use ChiselStage instead of chisel3.docs.emitSystemVerilog because we want layers printed here (obviously)
import circt.stage.ChiselStage
ChiselStage.emitSystemVerilog(new Foo, firtoolOpts=Array("-strip-debug-info", "-disable-all-randomization"))
```
