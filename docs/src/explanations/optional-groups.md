---
layout: docs
title:  "Optional Groups"
section: "chisel3"
---

# Optional Groups

Optional Groups are used to describe functionality of a Chisel module that a
user would like to _optionally_ include in the final design.  This feature can
be used to optionally include verification or debug logic that a user does not
want to have present in the final design.

Each optional group is broken into two pieces:

1. A _declaration_ of an optional group
1. One or more _definitions_ of an optional group inside a module

A _declaration_ of an optional group indicates that optional functionality may
exist in a Chisel circuit.  Declarations may be nested.  Declarations specify
the _convention_ that they use when lowering to Verilog.

To declare an optional group, extend the `chisel3.group.Declaration` abstract
class and specify a convention.  To declare a nested optional group, extend the
`chisel3.group.Declaration` abstract class inside another declaration.

The following example declares four optional groups:

```scala mdoc:silent
import chisel3.group.{Convention, Declaration}

object A extends Declaration(Convention.Bind) {
  object B extends Declaration(Convention.Bind) {
    object C extends Declaration(Convention.Bind)
  }
  object D extends Declaration(Convention.Bind)
}
```

A _definition_ of an optional group adds optional functionality to a module.
Each definition must refer to a declaration of an optional group.  Definitions
must match the nesting of their declarations.

To define an optional group, use the `chisel3.group` apply method referring to a
previously declared optional group inside a Chisel module.  An optional group
definition may use any Chisel or Scala value visible to its Scala lexical scope.

The following example defines four optional groups inside module `Foo` and
declares wires which are connected to values captured from visible lexical
scope:

```scala mdoc:silent
import chisel3._
import chisel3.group

class Foo extends RawModule {
  val port = IO(Input(Bool()))

  group(A) {
    val a = WireInit(port)
    group(A.B) {
      val b = WireInit(a)
      group(A.B.C) {
        val c = WireInit(b)
      }
    }
    group(A.D) {
      val d = WireInit(port ^ a)
    }
  }
}
```

## Conventions

Currently, there is only one supported convetion, `Bind`.  This will cause
optional group definitions to be lowered to Verilog modules that are
instantiated via the SystemVerilog `bind` mechanism.  The lowering to Verilog of
nested groups avoids illegal nested usage of `bind`.

More conventions may be supported in the future.

## Examples

### Design Verification Example

Consider a use case where a design or design verification engineer would like to
add some asserts and debug prints to a module.  The logic necessary for the
asserts and debug prints requires additional computation.  All of this code
should not be included in the final Verilog.  The engineer can use three
optional groups to do this.

There are three optional groups that emerge from this example:

1. A common `Verification` group
1. An `Assert` group nested under the `Verification` group
1. A `Debug` group also nested under the `Verification` group

The `Verification` group can be used to store common logic used by both the
`Assert` and `Debug` groups.  The latter two groups allow for separation of,
respectively, assertions from prints.

One way to write this in Scala is the following:

```scala mdoc:reset:silent
import chisel3._
import chisel3.group.{Convention, Declaration}

// All groups are declared here.  The Assert and Debug groups are nested under
// the Verification group.
object Verification extends group.Declaration(group.Convention.Bind) {
  object Assert extends group.Declaration(group.Convention.Bind)
  object Debug extends group.Declaration(group.Convention.Bind)
}

class Foo extends Module {
  val a = IO(Input(UInt(32.W)))
  val b = IO(Output(UInt(32.W)))

  b := a +% 1.U

  // This defines the `Verification` group inside Foo.
  group(Verification) {

    // Some common logic added here.  The input port `a` is "captured" and
    // used here.
    val a_d0 = RegNext(a)

    // This defines the `Assert` group.
    group(Verification.Assert) {
      chisel3.assert(a >= a_d0, "a must always increment")
    }

    // This defines the `Debug` group.
    group(Verification.Debug) {
      printf("a: %x, a_d0: %x", a, a_d0)
    }
  }

}

```

After compilation, this will produce three group include files with the
following filenames.  One file is created for each optional group:

1. `groups_Foo_Verification.sv`
1. `groups_Foo_Verification_Assert.sv`
1. `groups_Foo_Verification_Debug.sv`

A user can then include any combination of these files in their design to
include the optional functionality describe by the `Verification`, `Assert`, or
`Debug` groups.  The `Assert` and `Debug` bind files automatically include the
`Verification` bind file for the user.

#### Implementation Notes

_Note: the names of the modules and the names of any files that contain these
modules are FIRRTL compiler implementation defined!  The only guarantee is the
existence of the three group include files.  The information in this subsection
is for informational purposes to aid understanding._

In implementation, a FIRRTL compiler creates four Verilog modules for the
circuit above (one for `Foo` and one for each optional group definition in
module `Foo`):

1. `Foo`
1. `Foo_Verification`
1. `Foo_Verification_Assert`
1. `Foo_Verification_Debug`

These will typically be created in separate files with names that match the
modules, i.e., `Foo.sv`, `Foo_Verification.sv`, `Foo_Verification_Assert.sv`,
and `Foo_Verification_Debug.sv`.

The ports of each module created from an optional group definition will be
automatically determined based on what that group captured from outside the
group.  In the example above, the `Verification` group definition captured port
`a`.  Both the `Assert` and `Debug` group definitions captured `a` and `a_d0`.
Groups may be optimized to remove/add ports or to move logic into an optional
group.

#### Verilog Output

The complete Verilog output for this example is reproduced below:

```scala mdoc:verilog
import circt.stage.ChiselStage
ChiselStage.emitSystemVerilog(new Foo, firtoolOpts=Array("-strip-debug-info", "-disable-all-randomization"))
```
