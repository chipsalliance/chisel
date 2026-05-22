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
// Below is required for scala 3 migration
import chisel3.experimental.fromStringToStringParam
```

```scala mdoc:compile-only
class Foo extends RawModule {
  val myresult = IntrinsicExpr("MyIntrinsic", UInt(32.W), "STRING" -> "test")(3.U, 5.U)
}
```

## Debug type-info intrinsics

:::warning

This feature is experimental and subject to change.
See [Notes on reflection](#notes-on-reflection) below for limitations.

:::

Chisel can optionally emit a family of `circt_debug_*` intrinsics that carry
Chisel-level type and constructor-parameter metadata through to FIRRTL, so that
downstream tooling (waveform viewers, debuggers) can reconstruct Bundle, Vec
and `ChiselEnum` type names as well as module parameters that are lost during
conversion to plain FIRRTL.

The following intrinsics are emitted:

- `circt_debug_moduleinfo` - one per module, with `typeName` and a JSON-encoded
  `params` string describing primary constructor arguments.
- `circt_debug_var` - one per top-level signal, port or memory, with `typeName`
  and `name`.
- `circt_debug_subfield` - one per Bundle/Vec element, linked to its parent via
  a `parent` attribute.
- `circt_debug_enumdef` - one per `ChiselEnum` type (emitted once per circuit),
  listing all variants and their literal values.

Emission is opt-in and off by default. Enable it by passing
`--with-experimental-debug-intrinsics` on the command line, or by adding
`EmitDebugIntrinsicsAnnotation` to the annotation seq programmatically:

```scala mdoc:compile-only
import circt.stage.ChiselStage

class MyBundle extends Bundle {
  val a = UInt(8.W)
  val b = Bool()
}
class MyModule(val width: Int) extends Module {
  val io = IO(new Bundle {
    val in  = Input(new MyBundle)
    val out = Output(new MyBundle)
  })
  io.out := io.in
}

// Emit CHIRRTL with circt_debug_* intrinsics interleaved.
val chirrtl = ChiselStage.emitCHIRRTL(
  new MyModule(8),
  args = Array("--with-experimental-debug-intrinsics")
)
```

Under the hood this activates the `AddDebugIntrinsics` stage phase, which walks
the elaborated circuit and appends intrinsics as secret commands. Without the
annotation/flag the phase is a no-op.

### Notes on reflection

Constructor-parameter extraction uses Java/Scala reflection on user module and
Bundle classes. Two consequences worth knowing about:

- **JDK 17+** restricts reflective access to non-public members. The pass calls
  `setAccessible(true)` on `val`-accessor methods to read constructor argument
  values. On a locked-down JVM this may throw `InaccessibleObjectException`;
  the pass catches it and emits parameters without values rather than crashing
  the build. To get values back, run with
  `--add-opens java.base/<package>=ALL-UNNAMED` for the affected packages, or
  use `--with-experimental-debug-intrinsics` only in development builds.
- **Scala 3 ctor parameter names**: on Scala 3 the pass reads constructor
  parameter names via Java reflection (`Parameter.getName`). Scala 3 does not
  emit the JVM `MethodParameters` attribute by default, so `getName` returns
  synthetic `arg0`, `arg1`, ... names -- the pass logs a one-shot warning per
  affected class. There is currently no built-in scalac flag to fix this; the
  warning is informational, and intrinsics still emit parameter types and
  values, just without source-level names. Java sources can recover names by
  compiling with `javac -parameters`.
- **Nested constructor parameters**: The 256-character cap bounds the *output size*
  of `toString`, not its execution cost — a slow or allocation-heavy
  `toString` will slow or OOM the compile. Avoid expensive `toString`s in
  classes you expect to feed into debug metadata. Cycles in the object graph
  are detected via identity tracking and printed as `toString` rather than
  recursed into.
