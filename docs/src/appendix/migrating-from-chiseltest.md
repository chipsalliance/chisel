---
title: "Migrating from ChiselTest"
sidebar_position: 0
---

# Migrating from ChiselTest to ChiselSim

## Background

With the release of Chisel 5, Chisel moved off of the legacy [Scala FIRRTL Compiler (SFC)](https://github.com/chipsalliance/firrtl) to the MLIR FIRRTL Compiler (MFC), part of the [llvm/circt](https://github.com/llvm/circt) project.
After this release, the Scala FIRRTL Compiler was no longer maintained.
This change in underlying compiler technology has been a crucial piece in enabling the addition of many new features to Chisel, including linear-temporal logic (LTL) properties, Probes, and Layers.
Unfortunately, Chisel 3's testing library, [ChiselTest](https://github.com/ucb-bar/chiseltest), is built around the SFC, making it difficult to support ChiselTest in Chisel 5 and beyond.
_ChiselTest is not used or maintained by the core Chisel development team or their employers._

ChiselSim is the approved replacement for ChiselTest in Chisel 5 and beyond.
ChiselSim is maintained and used by the core Chisel development team.
This page describes how to migrate from ChiselTest to ChiselSim.

## Getting Started

The developers of ChiselTest have maintained some amount of compatibility between ChiselTest and newer versions of Chisel.
This relies on a forked version of the SFC.
Use of ChiselTest with Chisel 6 or later will prevent the usage of new Chisel 6 features.
It is not expected that new versions of Chisel will be compatible with the SFC.

We recommend using the latest minor version of Chisel 5 and ChiselTest 5 for migrating. At the time of writing this is Chisel v5.1.0 and ChiselTest v5.0.2, but please check for later versions.

## Migration

ChiselSim provides a minimal `peek`, `poke`, `expect`, and `step` API, similar to that of ChiselTest.
You use ChiselSim by importing it `import chisel3.simulator.EphemeralSimulator._`, and using its `simulate` method which is similar to ChiselTest's `test`.
At present, it does not have any integration with ScalaTest, so users should use any ScalaTest APIs directly.

For example, given a simple design (typically in `src/main/scala`):

```scala mdoc:silent
import chisel3._
class MyModule extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(16.W))
    val out = Output(UInt(16.W))
  })

  io.out := RegNext(io.in)
}
```

The legacy ChiselTest way to test this would be with a `ChiselScalatestTester` in `src/test/scala`:

<!-- This cannot be mdoc because we do not compile against chiseltest -->
```scala
import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class MyModuleSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "MyModule"
  it should "do something" in {
    test(new MyModule) { c =>
      c.io.in.poke(0.U)
      c.clock.step()
      c.io.out.expect(0.U)
      c.io.in.poke(42.U)
      c.clock.step()
      c.io.out.expect(42.U)
      println("Last output value : " + c.io.out.peek().litValue)
    }
  }
}
```

This can be rewritten using ChiselSim as follows:
```scala mdoc:silent
import chisel3._
import chisel3.simulator.EphemeralSimulator._
import org.scalatest.flatspec.AnyFlatSpec

class MyModuleSpec extends AnyFlatSpec {
  behavior of "MyModule"
  it should "do something" in {
    simulate(new MyModule) { c =>
      c.io.in.poke(0.U)
      c.clock.step()
      c.io.out.expect(0.U)
      c.io.in.poke(42.U)
      c.clock.step()
      c.io.out.expect(42.U)
      println("Last output value : " + c.io.out.peek().litValue)
    }
  }
}
```

For both ChiselTest and ChiselSim, you will typically run this with `sbt test` or some way of running tests.
The output from ChiselSim will look something like the following:

```scala mdoc
// This is how one can run a ScalaTest Spec manually, typically one would use "sbt test"
org.scalatest.nocolor.run(new MyModuleSpec)
```

ChiselSim also does not currently have any support for `fork`-`join`, so any tests using those constructs will need to be rewritten in a single-threaded manner.