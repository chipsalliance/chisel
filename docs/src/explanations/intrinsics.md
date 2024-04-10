---
layout: docs
title:  "Intrinsics"
section: "chisel3"
---

# Intrinsics

Chisel *Intrinsics* are used to express implementation defined functionality. 
Intrinsics provide a way for specific compilers to extend the capabilities of
the language in ways which are not implementable with library code.

Intrinsics will be typechecked by the implementation.  What intrinsics are 
available is documented by an implementation.

The `Intrinsic` and `IntrinsicExpr` can be used to create intrinsic statements
and expressions.

Modules defined as an `IntrinsicModule` will be instantiated as normal modules, 
but the intrinsic field communicates to the compiler what functionality to use to 
implement the module.  Implementations may not be actual modules, the module 
nature of intrinsics is merely for instantiation purposes.

### Parameterization

Parameters can be passed as an argument to the IntModule constructor.

### Intrinsic Expression Example

This following creates an intrinsic for the intrinsic named "MyIntrinsic".
It takes a parameter named "STRING" and has several inputs.

```scala mdoc:invisible
import chisel3._
```

```scala mdoc:compile-only
class Foo extends RawModule {
 val myresult = IntrinsicExpr("MyIntrinsic", Map("STRING" -> "test"), UInt(32.W))(3.U, 5.U)
}
```

### IntrinsicModule Example

This following creates an intrinsic module for the intrinsic named
"OtherIntrinsic".  It takes a parameter named "STRING" and has one bundle port.

```scala mdoc:invisible
import chisel3._
```

```scala mdoc:compile-only
import chisel3.experimental.IntrinsicModule

class ExampleIntrinsicModule(str: String) extends IntrinsicModule(
  "OtherIntrinsic",
  Map("STRING" -> str)) {
  val foo = IO(new Bundle() {
    val in = Input(UInt())
    val out = Output(UInt(32.W))
  })
}
```
