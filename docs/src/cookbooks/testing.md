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
./mill 'chisel[2.13.17].test.testOnly' chiselTests.ShiftRegistersSpec -- -DemitVcd=1
```

## How do I see what options a ChiselSim Scalatest test supports?

Pass `-Dhelp=1` to Scalatest, e.g.:

``` shell
./mill 'chisel[2.13.17].test.testOnly' chiselTests.ShiftRegistersSpec -- -Dhelp=1
```
