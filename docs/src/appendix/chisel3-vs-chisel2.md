---
layout: docs
title:  "Chisel3 vs. Chisel2"
section: "chisel3"
---

# Chisel3 vs Chisel2
```scala mdoc:invisible
import chisel3._
```

**Note** Chisel2 Compatibility Mode is entirely deprecated in 3.6, so this entire page is relevant only for 3.6 and earlier.

## Chisel2 Migration
For those moving from Chisel2, there were some backwards incompatible changes
and your RTL needs to be modified to work with Chisel3. The required
modifications are:

 - Wire declaration style:
```scala
  val wire = UInt(width = 15)
```
   becomes (in Chisel3):
```scala mdoc:compile-only
  val wire = Wire(UInt(15.W))
```

 - I/O declaration style:
```scala
   val done = Bool(OUTPUT)
```
   becomes (in Chisel3):
```scala mdoc:compile-only
   val wire = Output(Bool())
```

 - Sequential memories:
```scala mdoc:invisible
import chisel3._
val enable = Bool()
```

```scala
   val addr = Reg(UInt())
   val mem = Mem(UInt(8.W), 1024, seqRead = true)
   val dout = when(enable) { mem(addr) }
```
   becomes (in Chisel3):
```scala mdoc:compile-only
   val addr = UInt()
   val mem = SyncReadMem(1024, UInt(8.W))
   val dout = mem.read(addr, enable)
```

   Notice the address register is now internal to the SyncReadMem(), but the data
   will still return on the subsequent cycle.

 - Generating Verilog for a module:
```scala mdoc:invisible
import chisel3._
class Hello extends RawModule
```

```scala
   object Hello {
     def main(args: Array[String]): Unit = {
       chiselMain(Array("--backend", "v"), () => Module(new Hello()))
     }
   }
```
   becomes (in Chisel3):
```scala mdoc:compile-only
   import circt.stage.ChiselStage
   object Hello {
     def main(args: Array[String]): Unit = {
       ChiselStage.emitSystemVerilog(new Hello())
     }
   }
```

 - Package changes:
   - Chisel.log2Ceil -> chisel3.util.log2Ceil
   - BitPat
   - Decoupled is also in chisel3.util

Please refer to the [Chisel3 compatibility section](https://github.com/ucb-bar/chisel#chisel3)
for instructions on preparing your Chisel2 designs for Chisel3.

## Deprecated Usage
*  `Vec(Reg)` should be replaced with `Reg(Vec)`,
*  type-only vals (no associated data) must be wrapped in a `Wire()` if they will be the destination of a wiring operation (":=" or " < >"),
*  masked bit patterns ('b??') should be created using BitPat(), not UInt() or Bits(),
*  the `clone` method required for parameterized Bundles has been renamed `cloneType`,
*  the con and alt inputs to a Mux must be type-compatible - both signed or both unsigned,
*  bulk-connection to a node that has been procedurally assigned-to is illegal,
*  `!=` is deprecated, use `=/=` instead,
*  use `SyncReadMem(...)` instead of `Mem(..., seqRead)`,
*  use `SyncReadMem(n:Int, out: => T)` instead of `SyncReadMem(out: => T, n:Int)`,
*  use `SyncReadMem(...)` instead of `SeqMem(...)`,
*  use `Mem(n:Int, t:T)` instead of `Mem(out:T, n:Int)`,
*  use `Vec(n:Int, gen: => T)` instead of `Vec(gen: => T, n:Int)`,
*  module io's must be wrapped in `IO()`.
*  The methods `asInput`, `asOutput`, and `flip` should be replaced by the `Input()`, `Output()`, and `Flipped()` object apply methods.

## Unsupported constructs
*  `Mem(..., orderedWrites)` is no longer supported,
*  masked writes are only supported for `Mem[Vec[_]]`,
* Chisel3 Vecs must all have the same type, unlike with Chisel2. Use `MixedVec` (see [Bundles and Vecs](../explanations/bundles-and-vecs)) for Vecs where the elements are of different types.
*  connections between `UInt` and `SInt` are illegal.
*  the `Node` class and object no longer exist (the class should have been private in Chisel2)
*  `printf()` is defined in the Chisel object and produces simulation printf()'s.
To use the Scala `Predef.printf()`, you need to qualify it with `Predef`.
*  in Chisel2, bulk-connects `<>` with unconnected source components do not update connections from the unconnected components.
** In Chisel3, bulk-connects strictly adhere to last connection semantics and unconnected OUTPUTs will be connected to INPUTs resulting in the assignment of random values to those inputs.
* In Chisel3, adding hardware inside `BlackBox` for simulation is no longer supported. (#289)
* `ChiselError` is gone
  * Change `ChiselError.error("error msg")` to `throw new Error("error msg")`
  * Change `ChiselError.info("info msg")` to `println("info msg")`
* In Chisel3, subword assignments are not supported. [Alternative constructions exist](https://github.com/freechipsproject/chisel3/wiki/Cookbook#how-do-i-do-subword-assignment-assign-to-some-bits-in-a-uint) in Chisel3.

## Further changes
* The clock signal was renamed from `clk` to `clock` in Chisel3.
* Change `getWidth()` to `getWidth`

## Packaging
Chisel3 is implemented as several packages.
The core DSL is differentiated from utility or library classes and objects, testers, and interpreters.
The prime components of the Chisel3 front end (the DSL and library objects) are:
* coreMacros - source locators provide Chisel line numbers for `firrtl` detected errors,
* chiselFrontend - main DSL components,
* chisel3 - compiler driver, interface packages, compatibility layer.

Due to the wonders of `sbt`, you need only declare a dependency on the chisel3 package, and the others will be downloaded as required.

The `firrtl` compiler is distributed as a separate package, and release versions will also be downloaded automatically as required.
If you choose to integrate the compiler into your own toolchain, or you're working with the development (master) branch of chisel3, you should clone the [firrtl](https://github.com/ucb-bar/firrtl) repo
and follow the instructions for installing the `firrtl` compiler.

The testers in Chisel3 are distributed as a separate package.
If you intend to use them in your tests, you will either need to clone the [chisel-testers](https://github.com/ucb-bar/chisel-testers) repo
or declare a dependency on the published version of the package.
See the `build.sbt` file in either the [chisel-template](https://github.com/ucb-bar/chisel-template) or [chisel-tutorial](https://github.com/ucb-bar/chisel-tutorial)
repos for examples of the latter.

## Simulation
Chisel2 was capable of directly generating a `C++` simulation from the Chisel code, or a harness for use with a `vcs` simulation.
Chisel3 relies on [verilator](http://www.veripool.org/wiki/verilator) to generate the `C++` simulation from the Verilog output of `firrtl`.
See the [Chisel3 README](https://github.com/ucb-bar/chisel3) for directions on installing `verilator`.

## Compile Options and Front End Checks (Strict vs. NotStrict)
Chisel3 introduces a formal specification for hardware circuit graphs: FIRRTL,
and Chisel3 itself (the Scala library implementing the Chisel DSL), is a relatively thin front end that generates FIRRTL.
Since the `firrtl` parser needs to validate FIRRTL input, most of the checks that were performed in Chisel2 were eliminated
from the initial Chisel3 front end (the DRY principle).

However, this does impact the ability to provide detailed messages for error conditions that could be detected in the Chisel3
front end. The decision was made to optionally enable stricter error checking (for connections and the use of raw types versus
hardware objects), based on specific imports.
This allows designs to move from less strict front end checks (largely compatible with Chisel2), to stricter checking,
on a file by file basis, by adjusting specific import statements.

```scala mdoc:compile-only
    import chisel3.ExplicitCompileOptions.Strict
```

enables stricter connection and usage checks, while

```scala mdoc:compile-only
    import chisel3.ExplicitCompileOptions.NotStrict
```

defers these checks to the `firrtl` compiler.

By default, the `Chisel` compatibility layer, invoked by:

```scala
    import Chisel._
```

implicitly defines the compile options as `chisel3.core.ExplicitCompileOptions.NotStrict`

whereas the Chisel3 package, invoked by:

```scala
    import chisel3._
```

implicitly defines the compile options as `chisel3.core.ExplicitCompileOptions.Strict`

Again, these implicit compile options definitions may be overridden by explicit imports.

Currently, the specific error checks (found in [CompileOptions.scala](https://github.com/ucb-bar/chisel3/blob/master/chiselFrontend/src/main/scala/chisel3/core/CompileOptions.scala)) are:

```scala
    trait CompileOptions {
      // Should Bundle connections require a strict match of fields.
      // If true and the same fields aren't present in both source and sink, a MissingFieldException,
      // MissingLeftFieldException, or MissingRightFieldException will be thrown.
      val connectFieldsMustMatch: Boolean
      // When creating an object that takes a type argument, the argument must be unbound (a pure type).
      val declaredTypeMustBeUnbound: Boolean
      // Module IOs should be wrapped in an IO() to define their bindings before the reset of the module is defined.
      val requireIOWrap: Boolean
      // If a connection operator fails, don't try the connection with the operands (source and sink) reversed.
      val dontTryConnectionsSwapped: Boolean
      // If connection directionality is not explicit, do not use heuristics to attempt to determine it.
      val dontAssumeDirectionality: Boolean
      // Issue a deprecation warning if Data.{flip, asInput,asOutput} is used
      // instead of Flipped, Input, or Output.
      val deprecateOldDirectionMethods: Boolean
      // Check that referenced Data have actually been declared.
      val checkSynthesizable: Boolean
    }
```

`chisel3.core.ExplicitCompileOptions.Strict` sets all CompileOptions fields to true and
`chisel3.core.ExplicitCompileOptions.NotStrict` sets them all to false.
Clients are free to define their own settings for these options.
Examples may be found in the test [CompileOptionsSpec](https://github.com/ucb-bar/chisel3/blob/master/src/test/scala/chiselTests/CompileOptionsTest.scala)
