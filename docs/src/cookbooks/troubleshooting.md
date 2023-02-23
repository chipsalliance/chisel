---
layout: docs
title:  "Troubleshooting"
section: "chisel3"
---

# Troubleshooting


This page is a starting point for recording common and not so common problems in developing with Chisel3.  In particular, those situations where there is a work around that will keep you going.

### `type mismatch` specifying width/value of a `UInt`/`SInt`

*I have some old code that used to work correctly in chisel2 (and still does if I use the `import Chisel._` compatibility layer)
but causes a `type mismatch` error in straight chisel3:*

```scala mdoc:silent:fail
class TestBlock extends Module {
	val io = IO(new Bundle {
		val output = Output(UInt(width=3))
	})
}
```
*produces*
```bash
type mismatch;
[error]  found   : Int(3)
[error]  required: chisel3.internal.firrtl.Width
[error] 		val output = Output(UInt(width=3))
```

The single argument, multi-function object/constructors from chisel2 have been removed from chisel3.
It was felt these were too prone to error and made it difficult to diagnose error conditions in chisel3 code.

In chisel3, the single argument to the `UInt`/`SInt` object/constructor specifies the *width* and must be a `Width` type.
Although there are no automatic conversions from `Int` to `Width`, an `Int` may be converted to a `Width` by applying the `W` method to an `Int`.
In chisel3, the above code becomes:
```scala mdoc:silent
import chisel3._

class TestBlock extends Module {
	val io = IO(new Bundle {
		val output = Output(UInt(3.W))
	})
}
```
`UInt`/`SInt` literals may be created from an `Int` with the application of either the `U` or `S` method.

```scala mdoc:fail
UInt(42)
```

in chisel2, becomes
```scala mdoc:silent
42.U
```
in chisel3

A literal with a specific width is created by calling the `U` or `S` method with a `W` argument.
Use:
```scala mdoc:silent
1.S(8.W)
```
to create an 8-bit wide (signed) literal with value 1.


### `Dynamic index ... is too wide/narrow for extractee ...`?

If the index is too narrow you can use `.pad` to increase the width.
```scala mdoc:silent
import chisel3.util.log2Up

class TooNarrow(extracteeWidth: Int, indexWidth: Int) {
  val extractee = Wire(UInt(extracteeWidth.W))
  val index = Wire(UInt(indexWidth.W))
  extractee(index.pad(log2Up(extracteeWidth)))
}
```

If the index is too wide you can use a bit extract to select the correct bits.
```scala mdoc:silent
class TooWide(extracteeWidth: Int, indexWidth: Int) {
  val extractee = Wire(UInt(extracteeWidth.W))
  val index = Wire(UInt(indexWidth.W))
  extractee(index(log2Up(extracteeWidth) - 1, 0))
}
```

Or use both if you are working on a generator where the widths may be too wide or too narrow under different circumstances.
```scala mdoc:silent
class TooWideOrNarrow(extracteeWidth: Int, indexWidth: Int) {
  val extractee = Wire(UInt(extracteeWidth.W))
  val index = Wire(UInt(indexWidth.W))
  extractee(index.pad(log2Up(indexWidth))(log2Up(extracteeWidth) - 1, 0))
}
```
