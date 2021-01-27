# chisel-circt

[![Maven Central](https://img.shields.io/maven-central/v/com.sifive/chisel-circt_2.12)](https://maven-badges.herokuapp.com/maven-central/com.sifive/chisel-circt_2.12)
[![Sonatype Nexus (Snapshots)](https://img.shields.io/nexus/s/com.sifive/chisel-circt_2.12?server=https%3A%2F%2Foss.sonatype.org)](https://oss.sonatype.org/content/repositories/snapshots/com/sifive/chisel-circt_2.12/)
[![Javadoc](https://javadoc.io/badge2/com.sifive/chisel-circt_2.12/javadoc.svg)](https://javadoc.io/doc/com.sifive/chisel-circt_2.12)

# Compile Chisel using CIRCT/MLIR

This library provides a `ChiselStage`-like interface for compiling a Chisel circuit using the MLIR-based [llvm/circt](https://github.com/llvm/circt) project.

**CIRCT is a work in progress! Not all Chisel and FIRRTL features are supported!**

If you suspect a CIRCT bug or have questions, you can file an issue here, [post on Discourse](https://llvm.discourse.group/c/Projects-that-want-to-become-official-LLVM-Projects/circt/), or [file an issue on CIRCT](https://github.com/llvm/circt/issues/new/choose).

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

After CIRCT installation is complete, you should have `firtool` on your path.

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
