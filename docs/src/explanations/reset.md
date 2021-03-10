---
layout: docs
title:  "Reset"
section: "chisel3"
---

# Reset

```scala mdoc:invisible
import chisel3._

class Submodule extends Module
```

As of Chisel 3.2.0, Chisel 3 supports both synchronous and asynchronous reset,
meaning that it can natively emit both synchronous and asynchronously reset registers.

The type of register that is emitted is based on the type of the reset signal associated
with the register.

There are three types of reset that implement a common trait `Reset`:
* `Bool` - constructed with `Bool()`. Also known as "synchronous reset".
* `AsyncReset` - constructed with `AsyncReset()`. Also known as "asynchronous reset".
* `Reset` - constructed with `Reset()`. Also known as "abstract reset".

For implementation reasons, the concrete Scala type is `ResetType`. Stylistically we avoid `ResetType`, instead using the common trait `Reset`.

Registers with reset signals of type `Bool` are emitted as synchronous reset flops.
Registers with reset signals of type `AsyncReset` are emitted as asynchronouly reset flops.
Registers with reset signals of type `Reset` will have their reset type _inferred_ during FIRRTL compilation.

### Reset Inference

FIRRTL will infer a concrete type for any signals of type abstract `Reset`.
The rules are as follows:
1. An abstract `Reset` with only signals of type `AsyncReset`, abstract `Reset`, and `DontCare`
in both its fan-in and fan-out will infer to be of type `AsyncReset`
2. An abstract `Reset` with signals of both types `Bool` and `AsyncReset` in its fan-in and fan-out
is an error.
3. Otherwise, an abstract `Reset` will infer to type `Bool`.

You can think about (3) as the mirror of (1) replacing `AsyncReset` with `Bool` with the additional
rule that abstract `Reset`s with neither `AsyncReset` nor `Bool` in their fan-in and fan-out will
default to type `Bool`.
This "default" case is uncommon and implies that reset signal is ultimately driven by a `DontCare`.

### Implicit Reset

A `Module`'s `reset` is of type abstract `Reset`.
Prior to Chisel 3.2.0, the type of this field was `Bool`.
For backwards compatability, if the top-level module has an implicit reset, its type will default to `Bool`.

#### Setting Implicit Reset Type

_New in Chisel 3.3.0_

If you would like to set the reset type from within a Module (including the top-level `Module`),
rather than relying on _Reset Inference_, you can mixin one of the following traits:
* `RequireSyncReset` - sets the type of `reset` to `Bool`
* `RequireAsyncReset` - sets the type of `reset` to `AsyncReset`

For example:

```scala mdoc:silent
class MyAlwaysSyncResetModule extends Module with RequireSyncReset {
  val mySyncResetReg = RegInit(false.B) // reset is of type Bool
}
```

```scala mdoc:silent
class MyAlwaysAsyncResetModule extends Module with RequireAsyncReset {
  val myAsyncResetReg = RegInit(false.B) // reset is of type AsyncReset
}
```

**Note:** This sets the concrete type, but the Scala type will remain `Reset`, so casting may still be necessary.
This comes up most often when using a reset of type `Bool` in logic.


### Reset-Agnostic Code

The purpose of abstract `Reset` is to make it possible to design hardware that is agnostic to the
reset discipline used.
This enables code reuse for utilities and designs where the reset discipline does not matter to
the functionality of the block.

Consider the two example modules below which are agnostic to the type of reset used within them:

```scala mdoc:silent
class ResetAgnosticModule extends Module {
  val io = IO(new Bundle {
    val out = UInt(4.W)
  })
  val resetAgnosticReg = RegInit(0.U(4.W))
  resetAgnosticReg := resetAgnosticReg + 1.U
  io.out := resetAgnosticReg
}

class ResetAgnosticRawModule extends RawModule {
  val clk = IO(Input(Clock()))
  val rst = IO(Input(Reset()))
  val out = IO(Output(UInt(8.W)))

  val resetAgnosticReg = withClockAndReset(clk, rst)(RegInit(0.U(8.W)))
  resetAgnosticReg := resetAgnosticReg + 1.U
  out := resetAgnosticReg
}
```

These modules can be used in both synchronous and asynchronous reset domains.
Their reset types will be inferred based on the context within which they are used.

### Forcing Reset Type

You can set the type of a Module's implicit reset as described [above](#setting-implicit-reset-type).

You can also cast to force the concrete type of reset.
* `.asBool` will reinterpret a `Reset` as `Bool`
* `.asAsyncReset` will reinterpret a `Reset` as `AsyncReset`.

You can then use `withReset` to use a cast reset as the implicit reset.
See ["Multiple Clock Domains"](../explanations/multi-clock) for more information about `withReset`.


The following will make `myReg` as well as both `resetAgnosticReg`s synchronously reset:

```scala mdoc:silent
class ForcedSyncReset extends Module {
  // withReset's argument becomes the implicit reset in its scope
  withReset (reset.asBool) {
    val myReg = RegInit(0.U)
    val myModule = Module(new ResetAgnosticModule)

    // RawModules do not have implicit resets so withReset has no effect
    val myRawModule = Module(new ResetAgnosticRawModule)
    // We must drive the reset port manually
    myRawModule.rst := Module.reset // Module.reset grabs the current implicit reset
  }
}
```

The following will make `myReg` as well as both `resetAgnosticReg`s asynchronously reset:

```scala mdoc:silent
class ForcedAysncReset extends Module {
  // withReset's argument becomes the implicit reset in its scope
  withReset (reset.asAsyncReset){
    val myReg = RegInit(0.U)
    val myModule = Module(new ResetAgnosticModule) // myModule.reset is connected implicitly

    // RawModules do not have implicit resets so withReset has no effect
    val myRawModule = Module(new ResetAgnosticRawModule)
    // We must drive the reset port manually
    myRawModule.rst := Module.reset // Module.reset grabs the current implicit reset
  }
}
```

**Note:** such casts (`asBool` and `asAsyncReset`) are not checked by FIRRTL.
In doing such a cast, you as the designer are effectively telling the compiler
that you know what you are doing and to force the type as cast.

### Last-Connect Semantics

It is **not** legal to override the reset type using last-connect semantics
unless you are overriding a `DontCare`:

```scala mdoc:silent
class MyModule extends Module {
  val resetBool = Wire(Reset())
  resetBool := DontCare
  resetBool := false.B // this is fine
  withReset(resetBool) {
    val mySubmodule = Module(new Submodule())
  }
  resetBool := true.B // this is fine
  resetBool := false.B.asAsyncReset // this will error in FIRRTL
}
```
