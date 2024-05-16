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
  val myresult = IntrinsicExpr("MyIntrinsic", UInt(32.W), "STRING" -> "test")(3.U, 5.U)
}
```
