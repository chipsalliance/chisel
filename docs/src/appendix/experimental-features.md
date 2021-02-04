---
layout: docs
title:  "Experimental Features"
section: "chisel3"
---

Chisel has a number of new features that are worth checking out.  This page is an informal list of these features and projects.

### FixedPoint
FixedPoint numbers are basic *Data* type along side of UInt, SInt, etc.  Most common math and logic operations
are supported. Chisel allows both the width and binary point to be inferred by the Firrtl compiler which can simplify
circuit descriptions. See [FixedPointSpec](https://github.com/freechipsproject/chisel3/tree/master/src/test/scala/chiselTests/FixedPointSpec.scala)

### Module Variants
The standard Chisel *Module* requires a `val io = IO(...)`, the experimental package introduces several
new ways of defining Modules
- BaseModule: no contents, instantiable
- BlackBox extends BaseModule
- UserDefinedModule extends BaseModule: this module can contain Chisel RTL. No default clock or reset lines. No default IO. - User should be able to specify non-io ports, ideally multiple of them.
- ImplicitModule extends UserModule: has clock, reset, and io, essentially current Chisel Module.
- RawModule: will be the user-facing version of UserDefinedModule
- Module: type-aliases to ImplicitModule, the user-facing version of ImplicitModule.

### Bundle Literals

Bundle literals can be constructed via an experimental import:

```scala mdoc
import chisel3._
import chisel3.experimental.BundleLiterals._

class MyBundle extends Bundle {
  val a = UInt(8.W)
  val b = Bool()
}

class Example extends RawModule {
  val out = IO(Output(new MyBundle))
  out := (new MyBundle).Lit(_.a -> 8.U, _.b -> true.B)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example)
```

Partial specification is allowed, defaulting any unconnected fields to 0 (regardless of type).

```scala mdoc
class Example2 extends RawModule {
  val out = IO(Output(new MyBundle))
  out := (new MyBundle).Lit(_.b -> true.B)
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example2)
```

Bundle literals can also be nested arbitrarily.

```scala mdoc
class ChildBundle extends Bundle {
  val foo = UInt(8.W)
}

class ParentBundle extends Bundle {
  val a = UInt(8.W)
  val b = new ChildBundle
}

class Example3 extends RawModule {
  val out = IO(Output(new ParentBundle))
  out := (new ParentBundle).Lit(_.a -> 123.U, _.b -> (new ChildBundle).Lit(_.foo -> 42.U))
}
```

```scala mdoc:verilog
chisel3.stage.ChiselStage.emitVerilog(new Example3)
```

Vec literals are not yet supported.

### Interval Type

**Intervals** are a new experimental numeric type that comprises UInt, SInt and FixedPoint numbers.
It augments these types with range information, i.e. upper and lower numeric bounds.
This information can be used to exercise tighter programmatic control over the ultimate widths of
signals in the final circuit.  The **Firrtl** compiler can infer this range information based on
operations and earlier values in the circuit. Intervals support all the ordinary bit and arithmetic operations
associated with UInt, SInt, and FixedPoint and adds the following methods for manipulating the range of
a **source** Interval with the IntervalRange of **target** Interval

#### Clip -- Fit the value **source** into the IntervalRange of **target**, saturate if out of bounds
The clip method applied to an interval creates a new interval based on the argument to clip,
and constructs the necessary hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be pegged to either maximum or minimum of result range as appropriate.

> Generates necessary hardware to clip values, values greater than range are set to range.high, values lower than range are set to range min.

#### Wrap -- Fit the value **source** into the IntervalRange of **target**, wrapping around if out of bounds
The wrap method applied to an interval creates a new interval based on the argument to wrap,
and constructs the necessary
hardware so that the source Interval's value will be mapped into the new Interval.
Values that are outside the result range will be wrapped until they fall within the result range.

> Generates necessary hardware to wrap values, values greater than range are set to range.high, values lower than range are set to range min.

> Does not handle out of range values that are less than half the minimum or greater than twice maximum

#### Squeeze -- Fit the value **source** into the smallest IntervalRange based on source and target.
The squeeze method applied to an interval creates a new interval based on the argument to clip, the two ranges must overlap
behavior of squeeze with inputs outside of the produced range is undefined.

> Generates no hardware, strictly a sizing operation

##### Range combinations

| Condition | A.clip(B) | A.wrap(B) | A.squeeze(B) |
| --------- | --------------- | --------------- | --------------- |
| A === B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A contains B   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| B contains A   | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min < B min, A max in B  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A min in B, A max > B max  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  | max(Alo, Blo), min(Ahi, Bhi)  |
| A strictly less than B   | error               | error               | error               |
| A strictly greater than B   | error               | error               | error               |


#### Applying binary point operators to an Interval

Consider a Interval with a binary point of 3: aaa.bbb

| operation | after operation | binary point | lower | upper | meaning |
| --------- | --------------- | ------------ | ----- | ----- | ------- |
| setBinaryPoint(2) | aaa.bb |  2 | X | X  | set the precision |
| shiftLeftBinaryPoint(2) | a.aabbb |  5 | X | X  | increase the precision |
| shiftRighBinaryPoint(2) | aaaa.b |  1 | X | X  | reduce the precision |
