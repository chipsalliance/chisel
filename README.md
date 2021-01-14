# Compile Chisel using CIRCT

This library provides a `ChiselStage`-like interface for compiling Chisel using the MLIR-based [llvm/circt](https://github.com/llvm/circt) project.

**Note: this library and the CIRCT project are both works in progress!
A large number of features are not supported in CIRCT, yet!**

## Requirements

- Install `sbt` and create a base Chisel project
- Build CIRCT and add it's binaries to your path (`firtool` should be on your path)

## Example

This library provides a substitute for `chisel3.stage.ChiselStage` called `circt.stage.ChiselStage` which has similar behavior.
You can then use this to compile Chisel circuits to one of the MLIR dialects or go all the way to Verilog.

``` scala
import chisel3._

/* The following stage behaves like chisel3.stage.ChiselStage, but uses CIRCT */
import circt.stage.ChiselStage

class Foo extends RawModule {
  val a = IO(Input(Bool()))
  val b = IO(Output(Bool()))

  b := ~a
}

println(ChiselStage.emitSystemVerilog(new Foo))
```
