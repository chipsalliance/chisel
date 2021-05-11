---
layout: docs
title:  "Naming Cookbook"
section: "chisel3"
---

```scala mdoc:invisible
import chisel3._
import chisel3.experimental.prefix
import chisel3.experimental.noPrefix
import chisel3.stage.ChiselStage
```

### I still have _T signals, can this be fixed?

First check - is the compiler plugin properly enabled? Scalac plugins are enabled via the scalac option
`-Xplugin:<path/to/jar>`. You can check which compiler plugins are enabled by running `show Compile / scalacOptions` in
the sbt prompt.

If the plugin is enabled, these signals could be intermediate values which are consumed by either assertions or when
predicates. In these cases, the compiler plugin often can't find a good prefix for the generated intermediate signals.
We recommend you manually insert calls to `prefix` to fix these cases. We did this to Rocket Chip and saw huge benefits!

### I still see _GEN signals, can this be fixed?

`_GEN` signals are usually generated from the FIRRTL compiler, rather than the Chisel library. We are working on
renaming these signals with more context-dependent names, but it is a work in progress. Thanks for caring!

### My module names are super unstable - I change one thing and Queue_1 becomes Queue_42. Help!

This is the infamous `Queue` instability problem. In general, these cases are best solved at the source - the module
itself! If you overwrite `desiredName` to include parameter information (see the
[explanation](../explanations/naming#set-a-module-name) for more info), then this can avoid this problem permanantly.
We've done this with some Chisel utilities with great results!

### I want to add some hardware or assertions, but each time I do all the signal names get bumped!

This is the classic "ECO" problem, and we provide descriptions in [explanation](../explanations/naming). In short,
we recommend wrapping all additional logic in a prefix scope, which enables a unique namespace. This should prevent
name collisions, which are what triggers all those annoying signal name bumps!

### I want to force a signal (or instance) name to something, how do I do that?

Use the `.suggestName` method, which is on all classes which subtype `Data`.

### All this prefixing is annoying, how do I fix it?

You can use the `noPrefix { ... }` to strip the prefix from all signals generated in that scope.

```scala mdoc
class ExampleNoPrefix extends Module {
  val in = IO(Input(UInt(2.W)))
  val out = IO(Output(UInt()))

  val add = noPrefix { in + in + in }

  out := add
}
```
```scala mdoc:verilog
ChiselStage.emitVerilog(new ExampleNoPrefix)
```

### I am still not getting the name I want. For example, inlining an instance changes my name!

In cases where a FIRRTL transform renames a signal/instance, you can use the `forcename` API:

```scala mdoc
import chisel3.util.experimental.{forceName, InlineInstance}

class WrapperExample extends Module {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val inst = Module(new Wrapper)
  inst.in := in
  out := inst.out
}
class Wrapper extends Module with InlineInstance {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  val inst = Module(new MyLeaf)
  forceName(inst, "inst")
  inst.in := in
  out := inst.out
}
class MyLeaf extends Module {
  val in = IO(Input(UInt(3.W)))
  val out = IO(Output(UInt(3.W)))
  out := in
}
```
```scala mdoc:verilog
ChiselStage.emitVerilog(new WrapperExample)
```

This can be used to rename instances and non-aggregate typed signals.
