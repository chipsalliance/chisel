# chisel-circt

[![Maven Central](https://img.shields.io/maven-central/v/com.sifive/chisel-circt_2.13)](https://maven-badges.herokuapp.com/maven-central/com.sifive/chisel-circt_2.13)
[![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/com.sifive/chisel-circt_2.13?server=https%3A%2F%2Foss.sonatype.org)](https://oss.sonatype.org/content/repositories/snapshots/com/sifive/chisel-circt_2.13/)
[![Javadoc](https://javadoc.io/badge2/com.sifive/chisel-circt_2.13/javadoc.svg)](https://javadoc.io/doc/com.sifive/chisel-circt_2.13)

# Compile Chisel using CIRCT/MLIR

This library provides a `ChiselStage`-like interface for compiling a Chisel circuit using the MLIR-based FIRRTL Compiler (MFC) included in the [llvm/circt](https://github.com/llvm/circt) project.
This is an alternative to the Scala-based FIRRTL Compiler (SFC) that Chisel uses by default and is developed in [chipsalliance/firrtl](https://github.com/chipsalliance/firrtl).

**The MFC is a feature complete FIRRTL compiler, but does not support every annotation and custom transform-backed extension to Chisel.**

If you suspect a CIRCT bug or have questions, you can file an issue on this repository, [post on Discourse](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/), or [file an issue on CIRCT](https://github.com/llvm/circt/issues/new/choose).

## Setup

Include the following in your `build.sbt`.
See the badges above for latest release or snapshot version.

``` scala
libraryDependencies += "com.sifive" %% "chisel-circt" % "X.Y.Z"
```

Additionally, install CIRCT.
You can either:

1. Build and install from [source](https://github.com/llvm/circt)
2. Use a [nightly docker image](https://github.com/orgs/circt/packages/container/package/images%2Fcirct) and the [`firtool` script](https://github.com/circt/images/blob/trunk/circt/utils/firtool)

After CIRCT installation is complete, you need `firtool` (the tool provided with CIRCT to compile FIRRTL circuits) on your path so `chisel-circt` can use it.

### Base Project

Alternatively, a base project is provided in [sifive/chisel-circt-demo](https://github.com/sifive/chisel-circt-demo).

## Example

You can use `circt.stage.ChiselStage` *almost* exactly like `chsel3.stage.ChiselStage`.
E.g., the following will compile a simple module using CIRCT.

``` scala
import chisel3._

class Foo extends RawModule {
  val a = IO(Input(Bool()))
  val b = IO(Output(Bool()))

  b := ~a
}

/* Note: this is using circt.stage.ChiselStage */
val verilogString = circt.stage.ChiselStage.emitSystemVerilog(new Foo)

println(verilogString)
/** This will return:
  *
  * module Foo(
  *   input  a,
  *   output b);
  *
  *   assign b = ~a;
  * endmodule
  */
```
