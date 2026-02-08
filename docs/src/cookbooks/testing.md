---
sidebar_position: 6
---

# Testing Cookbook

import TOCInline from '@theme/TOCInline';

<TOCInline toc={toc} />

## How do I change the default testing directory?

Override the `buildDir` method.

The example below changes the testing directory to `test/`:

``` scala mdoc:reset:silent
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import java.nio.file.Paths
import org.scalatest.funspec.AnyFunSpec

class FooSpec extends FunSpec with ChiselSim {

  override def buildDir: Path = Paths.get("test")

}

```

## How do I enable waveforms for a simulation?

If using Scalatest and ChiselSim, pass the `-DemitVcd=1` argument to Scalatest, e.g.:

``` shell
./mill 'chisel[].test.testOnly' chiselTests.ShiftRegistersSpec -- -DemitVcd=1
```

## How do I enable Verilator coverage in svsim?

If you are configuring `svsim` directly, set Verilator backend coverage settings
at compile time:

```scala
import svsim.verilator

val backend = verilator.Backend.initializeFromProcessEnvironment()
val settings = verilator.Backend.CompilationSettings.default
  .withCoverageSettings(
    new verilator.Backend.CompilationSettings.CoverageSettings(
      line = true,
      toggle = true,
      user = true
    )
  )
```

This enables Verilator coverage instrumentation and writes `coverage.dat` at
the end of simulation. You can convert it to LCOV info with `verilator_coverage`
for downstream reporting tools.

## How do I see what options a ChiselSim Scalatest test supports?

Pass `-Dhelp=1` to Scalatest, e.g.:

``` shell
./mill 'chisel[].test.testOnly' chiselTests.ShiftRegistersSpec -- -Dhelp=1
```
