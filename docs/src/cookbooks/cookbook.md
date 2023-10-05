---
layout: docs
title:  "General Cookbook"
section: "chisel3"
---

# General Cookbook


Please note that these examples make use of [Chisel's scala-style printing](../explanations/printing#scala-style).

* Type Conversions
  * [How do I create a UInt from an instance of a Bundle?](#how-do-i-create-a-uint-from-an-instance-of-a-bundle)
  * [How do I create a Bundle from a UInt?](#how-do-i-create-a-bundle-from-a-uint)
  * [How can I tieoff a Bundle/Vec to 0?](#how-can-i-tieoff-a-bundlevec-to-0)
  * [How do I create a Vec of Bools from a UInt?](#how-do-i-create-a-vec-of-bools-from-a-uint)
  * [How do I create a UInt from a Vec of Bool?](#how-do-i-create-a-uint-from-a-vec-of-bool)
  * [How do I connect a subset of Bundle fields?](#how-do-i-connect-a-subset-of-bundle-fields)
* Vectors and Registers
  * [Can I make a 2D or 3D Vector?](#can-i-make-a-2D-or-3D-Vector)
  * [How do I create a Vector of Registers?](#how-do-i-create-a-vector-of-registers)
  * [How do I create a Reg of type Vec?](#how-do-i-create-a-reg-of-type-vec)
  * [How do I partially reset an Aggregate Reg?](#how-do-i-partially-reset-an-aggregate-reg)
* Bundles
  * [How do I deal with aliased Bundle fields?](#aliased-bundle-fields)
  * [How do I deal with the "unable to clone" error?](#bundle-unable-to-clone)
* [How do I create a finite state machine?](#how-do-i-create-a-finite-state-machine-fsm)
* [How do I unpack a value ("reverse concatenation") like in Verilog?](#how-do-i-unpack-a-value-reverse-concatenation-like-in-verilog)
* [How do I do subword assignment (assign to some bits in a UInt)?](#how-do-i-do-subword-assignment-assign-to-some-bits-in-a-uint)
* [How do I create an optional I/O?](#how-do-i-create-an-optional-io)
* [How do I create I/O without a prefix?](#how-do-i-create-io-without-a-prefix)
* [How do I minimize the number of bits used in an output vector](#how-do-i-minimize-the-number-of-bits-used-in-an-output-vector)
* [How do I resolve "Dynamic index ... is too wide/narrow for extractee ..."?](#dynamic-index-too-wide-narrow)
* Predictable Naming
  * [How do I get Chisel to name signals properly in blocks like when/withClockAndReset?](#how-do-i-get-chisel-to-name-signals-properly-in-blocks-like-whenwithclockandreset)
  * [How do I get Chisel to name the results of vector reads properly?](#how-do-i-get-chisel-to-name-the-results-of-vector-reads-properly)
  * [How can I dynamically set/parametrize the name of a module?](#how-can-i-dynamically-setparametrize-the-name-of-a-module)
* Directionality
  * [How do I strip directions from a bidirectional Bundle (or other Data)?](#how-do-i-strip-directions-from-a-bidirectional-bundle-or-other-data)

## Type Conversions

### How do I create a UInt from an instance of a Bundle?

Call [`asUInt`](https://www.chisel-lang.org/api/latest/chisel3/Bundle.html#asUInt:chisel3.UInt) on the [`Bundle`](https://www.chisel-lang.org/api/latest/chisel3/Bundle.html) instance.

```scala mdoc:silent:reset
import chisel3._

class MyBundle extends Bundle {
  val foo = UInt(4.W)
  val bar = UInt(4.W)
}

class Foo extends Module {
  val bundle = Wire(new MyBundle)
  bundle.foo := 0xc.U
  bundle.bar := 0x3.U
  val uint = bundle.asUInt
  printf(cf"$uint") // 195

  // Test
  assert(uint === 0xc3.U)
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

### How do I create a Bundle from a UInt?

Use the [`asTypeOf`](https://www.chisel-lang.org/api/latest/chisel3/UInt.html#asTypeOf[T%3C:chisel3.Data](that:T):T) method to reinterpret the [`UInt`](https://www.chisel-lang.org/api/latest/chisel3/UInt.html) as the type of the [`Bundle`](https://www.chisel-lang.org/api/latest/chisel3/Bundle.html).

```scala mdoc:silent:reset
import chisel3._

class MyBundle extends Bundle {
  val foo = UInt(4.W)
  val bar = UInt(4.W)
}

class Foo extends Module {
  val uint = 0xb4.U
  val bundle = uint.asTypeOf(new MyBundle)

  printf(cf"$bundle") // Bundle(foo -> 11, bar -> 4)

  // Test
  assert(bundle.foo === 0xb.U)
  assert(bundle.bar === 0x4.U)
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

### How can I tieoff a Bundle/Vec to 0?

You can use `asTypeOf` as above. If you don't want to worry about the type of the thing
you are tying off, you can use `chiselTypeOf`:

```scala mdoc:silent:reset
import chisel3._
import circt.stage.ChiselStage

class MyBundle extends Bundle {
  val foo = UInt(4.W)
  val bar = Vec(4, UInt(1.W))
}

class Foo(typ: MyBundle) extends Module {
  val bundleA = IO(Output(typ))
  val bundleB = IO(Output(typ))

  // typ is already a Chisel Data Type, so can use it directly here, but you
  // need to know that bundleA is of type typ
  bundleA := 0.U.asTypeOf(typ)

  // bundleB is a Hardware data IO(Output(...)) so need to call chiselTypeOf,
  // but this will work no matter the type of bundleB:
  bundleB := 0.U.asTypeOf(chiselTypeOf(bundleB))
}

ChiselStage.emitSystemVerilog(new Foo(new MyBundle))
```
### How do I create a Vec of Bools from a UInt?

Use [`VecInit`](https://www.chisel-lang.org/api/latest/chisel3/VecInit$.html) given a `Seq[Bool]` generated using the [`asBools`](https://www.chisel-lang.org/api/latest/chisel3/UInt.html#asBools:Seq[chisel3.Bool]) method.

```scala mdoc:silent:reset
import chisel3._

class Foo extends Module {
  val uint = 0xc.U
  val vec = VecInit(uint.asBools)

  printf(cf"$vec") // Vec(0, 0, 1, 1)

  // Test
  assert(vec(0) === false.B)
  assert(vec(1) === false.B)
  assert(vec(2) === true.B)
  assert(vec(3) === true.B)
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

### How do I create a UInt from a Vec of Bool?

Use the builtin function [`asUInt`](https://www.chisel-lang.org/api/latest/chisel3/Vec.html#asUInt:chisel3.UInt)

```scala mdoc:silent:reset
import chisel3._

class Foo extends Module {
  val vec = VecInit(true.B, false.B, true.B, true.B)
  val uint = vec.asUInt

  printf(cf"$uint") // 13

  // Test
  // (remember leftmost Bool in Vec is low order bit)
  assert(0xd.U === uint)

}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

### How do I connect a subset of Bundle fields?

See the [DataView cookbook](dataview#how-do-i-connect-a-subset-of-bundle-fields).

## Vectors and Registers

### Can I make a 2D or 3D Vector?

Yes. Using `VecInit` you can make Vectors that hold Vectors of Chisel types. Methods `fill` and `tabulate` make these multi-dimensional Vectors.

```scala mdoc:silent:reset
import chisel3._

class MyBundle extends Bundle {
  val foo = UInt(4.W)
  val bar = UInt(4.W)
}

class Foo extends Module {
  //2D Fill
  val twoDVec = VecInit.fill(2, 3)(5.U)
  //3D Fill
  val myBundle = Wire(new MyBundle)
  myBundle.foo := 0xc.U
  myBundle.bar := 0x3.U
  val threeDVec = VecInit.fill(1, 2, 3)(myBundle)
  assert(threeDVec(0)(0)(0).foo === 0xc.U && threeDVec(0)(0)(0).bar === 0x3.U)

  //2D Tabulate
  val indexTiedVec = VecInit.tabulate(2, 2){ (x, y) => (x + y).U }
  assert(indexTiedVec(0)(0) === 0.U)
  assert(indexTiedVec(0)(1) === 1.U)
  assert(indexTiedVec(1)(0) === 1.U)
  assert(indexTiedVec(1)(1) === 2.U)
  //3D Tabulate
  val indexTiedVec3D = VecInit.tabulate(2, 3, 4){ (x, y, z) => (x + y * z).U }
  assert(indexTiedVec3D(0)(0)(0) === 0.U)
  assert(indexTiedVec3D(1)(1)(1) === 2.U)
  assert(indexTiedVec3D(1)(1)(2) === 3.U)
  assert(indexTiedVec3D(1)(1)(3) === 4.U)
  assert(indexTiedVec3D(1)(2)(3) === 7.U)
}
```
```scala mdoc:invisible
// Hidden but will make sure this actually compiles
import circt.stage.ChiselStage

ChiselStage.emitSystemVerilog(new Foo)
```


### How do I create a Vector of Registers?

**Rule!  Use Reg of Vec not Vec of Reg!**

You create a [Reg of type Vec](#how-do-i-create-a-reg-of-type-vec). Because Vecs are a *type* (like `UInt`, `Bool`) rather than a *value*, we must bind the Vec to some concrete *value*.

### How do I create a Reg of type Vec?

For more information, the API Documentation for [`Vec`](https://www.chisel-lang.org/api/latest/chisel3/Vec.html) provides more information.

```scala mdoc:silent:reset
import chisel3._

class Foo extends Module {
  val regOfVec = Reg(Vec(4, UInt(32.W))) // Register of 32-bit UInts
  regOfVec(0) := 123.U                   // Assignments to elements of the Vec
  regOfVec(1) := 456.U
  regOfVec(2) := 789.U
  regOfVec(3) := regOfVec(0)

  // Reg of Vec of 32-bit UInts initialized to zero
  //   Note that Seq.fill constructs 4 32-bit UInt literals with the value 0
  //   VecInit(...) then constructs a Wire of these literals
  //   The Reg is then initialized to the value of the Wire (which gives it the same type)
  val initRegOfVec = RegInit(VecInit(Seq.fill(4)(0.U(32.W))))
}
```
```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```


### How do I partially reset an Aggregate Reg?

The easiest way is to use a partially-specified [Bundle Literal](#../appendix/experimental-features#bundle-literals)
or [Vec Literal](#../appendix/experimental-features#vec-literals) to match the type of the Reg.

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.BundleLiterals._

class MyBundle extends Bundle {
  val foo = UInt(8.W)
  val bar = UInt(8.W)
}

class MyModule extends Module {
  // Only .foo will be reset, .bar will have no reset value
  val reg = RegInit((new MyBundle).Lit(_.foo -> 123.U))
}
```

If your initial value is not a literal, or if you just prefer, you can use a
Wire as the initial value for the Reg. Simply connect fields to `DontCare` that
you do not wish to be reset.

```scala mdoc:silent
class MyModule2 extends Module {
  val reg = RegInit({
    // The wire could be constructed before the reg rather than in the RegInit scope,
    // but this style has nice lexical scoping behavior, keeping the Wire private
    val init = Wire(new MyBundle)
    init := DontCare // No fields will be reset
    init.foo := 123.U // Last connect override, .foo is reset
    init
  })
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new MyModule)
getVerilogString(new MyModule2)
```


## Bundles

### <a name="aliased-bundle-fields"></a> How do I deal with aliased Bundle fields?

```scala mdoc:invisible:reset
import chisel3._

class Top[T <: Data](gen: T) extends Module {
  val in = IO(Input(gen))
  val out = IO(Output(gen))
  out := in
}
```

Following the `gen` pattern when creating Bundles can result in some opaque error messages:

```scala mdoc
class AliasedBundle[T <: Data](gen: T) extends Bundle {
  val foo = gen
  val bar = gen
}
```

```scala mdoc:crash
getVerilogString(new Top(new AliasedBundle(UInt(8.W))))
```

This error is saying that fields `foo` and `bar` of `AliasedBundle` are the
exact same object in memory.
This is a problem for Chisel because we need to be able to distinguish uses of
`foo` and `bar` but cannot when they are referentially the same.

Note that the following example looks different but will give you exactly the same issue:

```scala mdoc
class AlsoAliasedBundle[T <: Data](val gen: T) extends Bundle {
                                // ^ This val makes `gen` a field, just like `foo`
  val foo = gen
}
```

By making `gen` a `val`, it becomes a public field of the `class`, just like `foo`.

```scala mdoc:crash
getVerilogString(new Top(new AlsoAliasedBundle(UInt(8.W))))
```

There are several ways to solve this issue with their own advantages and disadvantages.

#### 1. 0-arity function parameters

Instead of passing an object as a parameter, you can pass a 0-arity function (a function with no arguments):

```scala mdoc
class UsingAFunctionBundle[T <: Data](gen: () => T) extends Bundle {
  val foo = gen()
  val bar = gen()
}
```

Note that the type of `gen` is now `() => T`.
Because it is now a function and not a subtype of `Data`, you can safely make `gen` a `val` without
it becoming a hardware field of the `Bundle`.

Note that this also means you must pass `gen` as a function, for example:

```scala mdoc:silent
getVerilogString(new Top(new UsingAFunctionBundle(() => UInt(8.W))))
```

<a name="aliased-warning"></a> **Warning**: you must ensure that `gen` creates fresh objects rather than capturing an already constructed value:

```scala mdoc:crash
class MisusedFunctionArguments extends Module {
  // This usage is correct
  val in = IO(Input(new UsingAFunctionBundle(() => UInt(8.W))))

  // This usage is incorrect
  val fizz = UInt(8.W)
  val out = IO(Output(new UsingAFunctionBundle(() => fizz)))
}
getVerilogString(new MisusedFunctionArguments)
```
In the above example, value `fizz` and fields `foo` and `bar` of `out` are all the same object in memory.


#### 2. By-name function parameters

Functionally the same as (1) but with more subtle syntax, you can use [Scala by-name function parameters](https://docs.scala-lang.org/tour/by-name-parameters.html):

```scala mdoc
class UsingByNameParameters[T <: Data](gen: => T) extends Bundle {
  val foo = gen
  val bar = gen
}
```

With this usage, you do not include `() =>` when passing the argument:

```scala mdoc:silent
getVerilogString(new Top(new UsingByNameParameters(UInt(8.W))))
```

Note that as this is just syntactic sugar over (1), the [same warning applies](#aliased-warning).

#### 3. Directioned Bundle fields

You can alternatively wrap the fields with `Output(...)`, which creates fresh instances of the passed argument.
Chisel treats `Output` as the "default direction" so if all fields are outputs, the `Bundle` is functionally equivalent to a `Bundle` with no directioned fields.

```scala mdoc
class DirectionedBundle[T <: Data](gen: T) extends Bundle {
  val foo = Output(gen)
  val bar = Output(gen)
}
```

```scala mdoc:invisible
getVerilogString(new Top(new DirectionedBundle(UInt(8.W))))
```

This approach is admittedly a little ugly and may mislead others reading the code because it implies that this Bundle is intended to be used as an `Output`.

#### 4. Call `.cloneType` directly

You can also just call `.cloneType` on your `gen` argument directly.
While we try to hide this implementation detail from the user, `.cloneType` is the mechanism by which Chisel creates fresh instances of `Data` objects:

```scala mdoc
class UsingCloneTypeBundle[T <: Data](gen: T) extends Bundle {
  val foo = gen.cloneType
  val bar = gen.cloneType
}
```

```scala mdoc:invisible
getVerilogString(new Top(new UsingCloneTypeBundle(UInt(8.W))))
```

### <a name="bundle-unable-to-clone"></a> How do I deal with the "unable to clone" error?

Most Chisel objects need to be cloned in order to differentiate between the
software representation of the bundle field from its "bound" hardware
representation, where "binding" is the process of generating a hardware
component. For Bundle fields, this cloning is supposed to happen automatically
with a compiler plugin.

In some cases though, the plugin may not be able to clone the Bundle fields. The
most common case for when this happens is when the `chisel3.Data` part of the
Bundle field is nested inside some other data structure and the compiler plugin
is unable to figure out how to clone the entire structure. It is best to avoid
such nested structures.

There are a few ways around this issue - you can try wrapping the problematic
fields in Input(...), Output(...), or Flipped(...) if appropriate. You can also
try manually cloning each field in the Bundle using the `chiselTypeClone` method
in `chisel3.reflect.DataMirror`. Here's an example with the Bundle whose fields
won't get cloned:

```scala mdoc:invisible
import chisel3._
import scala.collection.immutable.ListMap
```

```scala mdoc:crash
class CustomBundleBroken(elts: (String, Data)*) extends Record {
  val elements = ListMap(elts: _*)

  def apply(elt: String): Data = elements(elt)
}

class NewModule extends Module {
  val out = Output(UInt(8.W))
  val recordType = new CustomBundleBroken("fizz" -> UInt(16.W), "buzz" -> UInt(16.W))
  val record = Wire(recordType)
  val uint = record.asUInt
  val record2 = uint.asTypeOf(recordType)
  out := record
}
getVerilogString(new NewModule)
```

You can use `chiselTypeClone` to clone the elements as:


```scala mdoc
import chisel3.reflect.DataMirror
import chisel3.experimental.requireIsChiselType

class CustomBundleFixed(elts: (String, Data)*) extends Record {
  val elements = ListMap(elts.map {
    case (field, elt) =>
      requireIsChiselType(elt)
      field -> DataMirror.internal.chiselTypeClone(elt)
  }: _*)

  def apply(elt: String): Data = elements(elt)
}
```

### How do I create a finite state machine (FSM)?

The advised way is to use `ChiselEnum` to construct enumerated types representing the state of the FSM.
State transitions are then handled with `switch`/`is` and `when`/`.elsewhen`/`.otherwise`.

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.{switch, is}

object DetectTwoOnes {
  object State extends ChiselEnum {
    val sNone, sOne1, sTwo1s = Value
  }
}

/* This FSM detects two 1's one after the other */
class DetectTwoOnes extends Module {
  import DetectTwoOnes.State
  import DetectTwoOnes.State._

  val io = IO(new Bundle {
    val in = Input(Bool())
    val out = Output(Bool())
    val state = Output(State())
  })

  val state = RegInit(sNone)

  io.out := (state === sTwo1s)
  io.state := state

  switch (state) {
    is (sNone) {
      when (io.in) {
        state := sOne1
      }
    }
    is (sOne1) {
      when (io.in) {
        state := sTwo1s
      } .otherwise {
        state := sNone
      }
    }
    is (sTwo1s) {
      when (!io.in) {
        state := sNone
      }
    }
  }
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new DetectTwoOnes)
```

Note: the `is` statement can take multiple conditions e.g. `is (sTwo1s, sOne1) { ... }`.

### How do I unpack a value ("reverse concatenation") like in Verilog?

In Verilog, you can do something like the following which will unpack a the value `z`:

```verilog
wire [1:0] a;
wire [3:0] b;
wire [2:0] c;
wire [8:0] z = [...];
assign {a,b,c} = z;
```

Unpacking often corresponds to reinterpreting an unstructured data type as a structured data type.
Frequently, this structured type is used prolifically in the design, and has been declared as in the following example:

```scala mdoc:silent:reset
import chisel3._

class MyBundle extends Bundle {
  val a = UInt(2.W)
  val b = UInt(4.W)
  val c = UInt(3.W)
}
```

The easiest way to accomplish this in Chisel would be:

```scala mdoc:silent
class Foo extends Module {
  val z = Wire(UInt(9.W))
  z := DontCare // This is a dummy connection
  val unpacked = z.asTypeOf(new MyBundle)
  printf("%d", unpacked.a)
  printf("%d", unpacked.b)
  printf("%d", unpacked.c)
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

If you **really** need to do this for a one-off case (Think thrice! It is likely you can better structure the code using bundles), then rocket-chip has a [Split utility](https://github.com/freechipsproject/rocket-chip/blob/723af5e6b69e07b5f94c46269a208a8d65e9d73b/src/main/scala/util/Misc.scala#L140) which can accomplish this.

### How do I do subword assignment (assign to some bits in a UInt)?

You may try to do something like the following where you want to assign only some bits of a Chisel type.
Below, the left-hand side connection to `io.out(0)` is not allowed.

```scala mdoc:silent:reset
import chisel3._
import circt.stage.ChiselStage

class Foo extends Module {
  val io = IO(new Bundle {
    val bit = Input(Bool())
    val out = Output(UInt(10.W))
  })
  io.out(0) := io.bit
}
```

If you try to compile this, you will get an error.
```scala mdoc:crash
getVerilogString(new Foo)
```

Chisel3 *does not support subword assignment*.
The reason for this is that subword assignment generally hints at a better abstraction with an aggregate/structured types, i.e., a `Bundle` or a `Vec`.

If you must express it this way, one approach is to blast your `UInt` to a `Vec` of `Bool` and back:

```scala mdoc:silent:reset
import chisel3._

class Foo extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(10.W))
    val bit = Input(Bool())
    val out = Output(UInt(10.W))
  })
  val bools = VecInit(io.in.asBools)
  bools(0) := io.bit
  io.out := bools.asUInt
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new Foo)
```

### How do I create an optional I/O?

The following example is a module which includes the optional port `out2` only if the given parameter is `true`.

```scala mdoc:silent:reset
import chisel3._

class ModuleWithOptionalIOs(flag: Boolean) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(12.W))
    val out = Output(UInt(12.W))
    val out2 = if (flag) Some(Output(UInt(12.W))) else None
  })

  io.out := io.in
  if (flag) {
    io.out2.get := io.in
  }
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new ModuleWithOptionalIOs(true))
```

The following is an example where an entire `IO` is optional:

```scala mdoc:silent:reset
import chisel3._

class ModuleWithOptionalIO(flag: Boolean) extends Module {
  val in = if (flag) Some(IO(Input(Bool()))) else None
  val out = IO(Output(Bool()))

  out := in.getOrElse(false.B)
}
```

```scala mdoc:invisible
// Hidden but will make sure this actually compiles
getVerilogString(new ModuleWithOptionalIO(true))
```

### How do I create I/O without a prefix?

In most cases, you can simply call `IO` multiple times:

```scala mdoc:silent:reset
import chisel3._

class MyModule extends Module {
  val in = IO(Input(UInt(8.W)))
  val out = IO(Output(UInt(8.W)))

  out := in +% 1.U
}
```

```scala mdoc:verilog
getVerilogString(new MyModule)
```

If you have a `Bundle` from which you would like to create ports without the
normal `val` prefix, you can use `FlatIO`:

```scala mdoc:silent:reset
import chisel3._
import chisel3.experimental.FlatIO

class MyBundle extends Bundle {
  val foo = Input(UInt(8.W))
  val bar = Output(UInt(8.W))
}

class MyModule extends Module {
  val io = FlatIO(new MyBundle)

  io.bar := io.foo +% 1.U
}
```

Note that `io_` is nowhere to be seen!

```scala mdoc:verilog
getVerilogString(new MyModule)
```

### How do I minimize the number of bits used in an output vector?

Use inferred width and a `Seq` instead of a `Vec`:

Consider:

```scala mdoc:silent:reset
import chisel3._

// Count the number of set bits up to and including each bit position
class CountBits(width: Int) extends Module {
  val bits = IO(Input(UInt(width.W)))
  val countSequence = Seq.tabulate(width)(i => IO(Output(UInt())))
  val countVector = IO(Output(Vec(width, UInt())))
  countSequence.zipWithIndex.foreach { case (port, i) =>
    port := util.PopCount(bits(i, 0))
  }
  countVector := countSequence
}
```

Unlike `Vecs` which represent a singular Chisel type and must have the same width for every element,
`Seq` is a purely Scala construct, so their elements are independent from the perspective of Chisel and can have different widths.

```scala mdoc:verilog
circt.stage.ChiselStage.emitSystemVerilog(new CountBits(4))
  // remove the body of the module by removing everything after ');'
  .split("\\);")
  .head + ");\n"
```

### <a id="dynamic-index-too-wide-narrow" /> How do I resolve "Dynamic index ... is too wide/narrow for extractee ..."?


Chisel will warn if a dynamic index is not the correctly-sized width for indexing a Vec or UInt.
"Correctly-sized" means that the width of the index should be the log2 of the size of the indexee.
If the indexee is a non-power-of-2 size, use the ceiling of the log2 result.

```scala mdoc:invisible:reset
import chisel3._
// Some other test is clobbering the global Logger which breaks the warnings below
// Setting the output stream to the Console fixes the issue
logger.Logger.setConsole()
// Helper to throw away return value so it doesn't show up in mdoc
def compile(gen: => chisel3.RawModule): Unit = {
  circt.stage.ChiselStage.emitCHIRRTL(gen)
}
```

When the index does not have enough bits to address all entries or bits in the extractee, you can `.pad` the index to increase the width.

```scala mdoc
class TooNarrow extends RawModule {
  val extractee = Wire(UInt(7.W))
  val index = Wire(UInt(2.W))
  extractee(index)
}
compile(new TooNarrow)
```

This can be fixed with `pad`:

```scala mdoc
class TooNarrowFixed extends RawModule {
  val extractee = Wire(UInt(7.W))
  val index = Wire(UInt(2.W))
  extractee(index.pad(3))
}
compile(new TooNarrowFixed)
```

#### Use bit extraction when the index is too wide

```scala mdoc
class TooWide extends RawModule {
  val extractee = Wire(Vec(8, UInt(32.W)))
  val index = Wire(UInt(4.W))
  extractee(index)
}
compile(new TooWide)
```

This can be fixed with bit extraction:

```scala mdoc
class TooWideFixed extends RawModule {
  val extractee = Wire(Vec(8, UInt(32.W)))
  val index = Wire(UInt(4.W))
  extractee(index(2, 0))
}
compile(new TooWideFixed)
```

Note that size 1 `Vecs` and `UInts` should be indexed by a zero-width `UInt`:

```scala mdoc
class SizeOneVec extends RawModule {
  val extractee = Wire(Vec(1, UInt(32.W)))
  val index = Wire(UInt(0.W))
  extractee(index)
}
compile(new SizeOneVec)
```

Because `pad` only pads if the desired width is less than the current width of the argument,
you can use `pad` in conjunction with bit extraction when the widths may be too wide or too
narrow under different circumstances

```scala mdoc
import chisel3.util.log2Ceil
class TooWideOrNarrow(extracteeSize: Int, indexWidth: Int) extends Module {
  val extractee = Wire(Vec(extracteeSize, UInt(8.W)))
  val index = Wire(UInt(indexWidth.W))
  val correctWidth = log2Ceil(extracteeSize)
  extractee(index.pad(correctWidth)(correctWidth - 1, 0))
}
compile(new TooWideOrNarrow(8, 2))
compile(new TooWideOrNarrow(8, 4))
```

Another option for dynamic bit selection of `UInts` (but not `Vec` dynamic indexing) is to do a dynamic
right shift of the extractee by the index and then just bit select a single bit:
```scala mdoc
class TooWideOrNarrowUInt(extracteeSize: Int, indexWidth: Int) extends Module {
  val extractee = Wire(UInt(extracteeSize.W))
  val index = Wire(UInt(indexWidth.W))
  (extractee >> index)(0)
}
compile(new TooWideOrNarrowUInt(8, 2))
compile(new TooWideOrNarrowUInt(8, 4))
```

## Predictable Naming

### How do I get Chisel to name signals properly in blocks like when/withClockAndReset?

Use the compiler plugin, and check out the [Naming Cookbook](naming) if that still does not do what you want.

### How do I get Chisel to name the results of vector reads properly?
Currently, name information is lost when using dynamic indexing. For example:
```scala mdoc:silent:reset
import chisel3._

class Foo extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(4, Bool()))
    val idx = Input(UInt(2.W))
    val en = Input(Bool())
    val out = Output(Bool())
  })

  val x = io.in(io.idx)
  val y = x && io.en
  io.out := y
}
```

The above code loses the `x` name, instead using `_GEN_3` (the other `_GEN_*` signals are expected).

{% raw %}
```scala mdoc:verilog
getVerilogString(new Foo)
```
{% endraw %}

This can be worked around by creating a wire and connecting the dynamic index to the wire:
```scala
val x = WireInit(io.in(io.idx))
```

```scala mdoc:invisible
class Foo2 extends Module {
  val io = IO(new Bundle {
    val in = Input(Vec(4, Bool()))
    val idx = Input(UInt(2.W))
    val en = Input(Bool())
    val out = Output(Bool())
  })

  val x = WireInit(io.in(io.idx))
  val y = x && io.en
  io.out := y
}
```

Which produces:
{% raw %}
```scala mdoc:verilog
getVerilogString(new Foo2)
```
{% endraw %}

### How can I dynamically set/parametrize the name of a module?

You can override the `desiredName` function. This works with normal Chisel modules and `BlackBox`es. Example:

```scala mdoc:silent:reset
import chisel3._

class Coffee extends BlackBox {
    val io = IO(new Bundle {
        val I = Input(UInt(32.W))
        val O = Output(UInt(32.W))
    })
    override def desiredName = "Tea"
}

class Salt extends Module {
    val io = IO(new Bundle {})
    val drink = Module(new Coffee)
    override def desiredName = "SodiumMonochloride"

    drink.io.I := 42.U
}
```

Elaborating the Chisel module `Salt` yields our "desired names" for `Salt` and `Coffee` in the output Verilog:

```scala mdoc:verilog
getVerilogString(new Salt)
```

## Directionality

### How do I strip directions from a bidirectional Bundle (or other Data)?

Given a bidirectional port like a `Decoupled`, you will get an error if you try to connect it directly
to a register:

```scala mdoc:silent:reset
import chisel3._
import circt.stage.ChiselStage
import chisel3.util.Decoupled
class BadRegConnect extends Module {
  val io = IO(new Bundle {
    val enq = Decoupled(UInt(8.W))
  })

  val monitor = Reg(chiselTypeOf(io.enq))
  monitor := io.enq
}
```

```scala mdoc:crash
ChiselStage.emitSystemVerilog(new BadRegConnect)
```

While there is no construct to "strip direction" in Chisel3, wrapping a type in `Output(...)`
(the default direction in Chisel3) will
set all of the individual elements to output direction.
This will have the desired result when used to construct a Register:

```scala mdoc:silent:reset
import chisel3._
import circt.stage.ChiselStage
import chisel3.util.Decoupled
class CoercedRegConnect extends Module {
  val io = IO(new Bundle {
    val enq = Flipped(Decoupled(UInt(8.W)))
  })

  // Make a Reg which contains all of the bundle's signals, regardless of their directionality
  val monitor = Reg(Output(chiselTypeOf(io.enq)))
  // Even though io.enq is bidirectional, := will drive all fields of monitor with the fields of io.enq
  monitor := io.enq
}
```

<!-- Just make sure it actually works -->
```scala mdoc:invisible
ChiselStage.emitSystemVerilog(new CoercedRegConnect {
  // Provide default connections that would just muddy the example
  io.enq.ready := true.B
  // dontTouch so that it shows up in the Verilog
  dontTouch(monitor)
})
```
