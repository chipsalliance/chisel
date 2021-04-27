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
