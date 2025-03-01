---
layout: docs
title:  "Testing"
section: "chisel3"
---

# Testing

Chisel provides several packages for testing generators with different
strategies.

The primary testing strategy is simulation.  This is done using _ChiselSim_, a
library for simulating Chisel-generated SystemVerilog on different simulators.

An alternative, complementary testing strategy is to directly inspect the
SystemVerilog or FIRRTL text that a Chisel generator produces.  This is done
using [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html).

:::tip

The apprpriate testing strategy will depend on what you are trying to test.  It
is likely that you will want a mix of different strategies.

:::

Both ChiselSim and FileCheck are provided as packages inside Chisel.
Subsections below describe these packages and their use.

## ChiselSim

ChiselSim provides a number of methods that you can use to run simulations and
provide stimulus to Chisel modules being tested.  To use ChiselSim, mix-in one
of the following two traits into a class:

- `chisel3.simulator.ChiselSim`
- `chisel3.simulator.scalatest.ChiselSim`

Both traits provide the same methods.  The latter trait provides tighter
integration with [ScalaTest](https://www.scalatest.org/) and will cause test
results to be placed in a directory structure derived from ScalaTest test names
for easy user inspection.

### Simulation APIs

ChiselSim provides two simulation APIs for running simulations.  These are:

- `simulate`
- `simulateRaw`

The former may only be used with `Module`s or their subtypes.  The latter may
only be used with `RawModule`s or their subtypes.

Thd difference between them is that `simulate` will put the module through an
initialization procedure before user stimulus is applied.

Conversely, `simulateRaw` will apply no initialization procedure and it is up to
the user to provide suitable reset stimulus.

:::info

The reason why `simulate` can define a reset procedure is because a `Module` has
a defined clock and reset port.  Because of this, a common pattern when working
with ChiselSim is to wrap your design under test in a test harness that is a
`Module`.  The test harness will be provided with the initialization stimulus
and any more complicated stimulus (e.g., multiple clocks) can be derived inside
the test harness.

:::

Simulation APIs take user provided stimulus and apply it to the module.  Some
useful stimulus is provided in the `chisel3.simulator.stimulus` package.  For
example, the `RunUntilFinished` stimulus will toggle a `Module`'s clock for a
number of cycles and throw an exception if the module does net execute a
`chisel3.stop` before that number of clock cycles has elapsed.

For more information see the [Chisel API
documentation](https://www.chisel-lang.org/api) for
`chisel3.simulator.SimulatorAPI`.

### Peek/Poke APIs

ChiselSim provides basic "peek", "poke", and "expect" APIs for providing simple
stimulus to Chisel modules.  This API is implemented as [extension
methods](https://en.wikipedia.org/wiki/Extension_method) on Chisel types, like
`Data`.  This means that the ports of your design under test have _new_ methods
defined on them that can be used to drive stimulus.

These APIs are summarized below:

- `poke` sets a value on a port
- `peek` reads a value on a port
- `expect` reads a value on a port and asserts that it is equal another value
- `step` toggles a clock for a number of cycles
- `stepUntil` toggles a clock until a condition occurs on another port

For more information see the [Chisel API
documentation](https://www.chisel-lang.org/api) for
`chisel3.simulator.PeekPokeAPI`.

### Example

The example below shows a basic usage of ChiselSim inside ScalaTest.  This shows
a single test suite, `ChiselSimExample`.  To gain access to ChiselSim methods,
the `ChiselSim` trait is mixed in.  A [testing
style](https://www.scalatest.org/user_guide/selecting_a_style) is chosen and
"should" matches are added to provide a more natural language way of writing
tests.

In the test, module `Foo` is tested using custom stimulus.  Module `Bar` is
tested using pre-defined stimulus.  Both tests, as written, will pass.


```scala mdoc:silent:reset
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.Counter
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ChiselSimExample extends AnyFlatSpec with Matchers with ChiselSim {

  class Foo extends Module {
    val a, b = IO(Input(UInt(8.W)))
    val c = IO(Output(chiselTypeOf(a)))

    private val r = Reg(chiselTypeOf(a))

    r :<= a +% b
    c :<= r
  }

  behavior of "Baz"

  it should "add two numbers" in {

    simulate(new Foo) { foo =>
      // Poke different values on the two input ports.
      foo.a.poke(1)
      foo.b.poke(2)

      // Step the clock by one cycle.
      foo.clock.step(1)

      // Expect that the sum of the two inputs is on the output port.
      foo.c.expect(3)
    }

  }

  class Bar extends Module {

    val (_, done) = Counter(true.B, 10)

    when (done) {
      stop()
    }

  }

  behavior of "Bar"

  it should "terminate before 11 cycles have elapsed" in {

    simulate(new Bar)(RunUntilFinished(11))

  }

}

```

## FileCheck

Sometimes, it is sufficient to directly inspect the result of a generator.  This
testing strategy is particularly relevent if you are trying to create very
specific Verilog structures or to guarantee exact naming of specific constructs.

While simple testing can be done with string comparisons, this is often
insufficient as it is necessary to both have a mixture of regular expression
captures and ordering of specific lines.  For this, Chisel provides a native way
to write [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests.

:::info

FileCheck is a tool used extensively in the testing of compilers in the LLVM
ecosystem.  [CIRCT](https://github.com/llvm/circt), the compiler that converts
the FIRRTL that Chisel produces into SystemVerilog, makes heavy use of FileCheck
for its own testing.

:::

Like with ChiselSim, two different traits are provided for writing FileCheck
tests:

- `chisel3.testing.FileCheck`
- `chisel3.testing.scalatest.FileCheck`

Both provide the same APIs, but the latter will write intermediary files to
directories derived from ScalaTest suite and scope names.

Presently, only one FileCheck API is provided: `fileCheck`.  This API is
implemented as an extension method on `String` and takes two arguments: (1) a
list of arguments to FileCheck and (2) a string that contains an inline
FileCheck test to run.  Both the input string and the check string will be
written to disk and preserved on failure so that you can rerun them manually if
needed.

If the `fileCheck` method succeeds, nothing is returned.  If it fails, it will
throw an exception indicating why it failed and verbose information aobut where
an expected string did not match.

For more information on the API see the [Chisel API
documentation](https://www.chisel-lang.org/api) for `chisel3.testing.FileCheck`.
For more information on FileCheck and its usage see the [FileCheck
documentation](https://llvm.org/docs/CommandGuide/FileCheck.html).

### Example

The example below shows a FileCheck test that checks that a module has a
specific name and that it has some expected content inside it.  Specifically,
this test is checking that constant propagation happens as expected.  As
written, this test will pass.

```scala mdoc:silent:reset
import chisel3._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class FileCheckExample extends AnyFlatSpec with Matchers with FileCheck {

  class Baz extends RawModule {

    val out = IO(Output(UInt(32.W)))

    out :<= 1.U(32.W) + 3.U(32.W)

  }

  behavior of "Foo"

  it should "simplify the constant computation in its body" in {

    ChiselStage.emitSystemVerilog(new Baz).fileCheck()(
      """|CHECK:      module Baz(
         |CHECK-NEXT:   output [31:0] out
         |CHECK:        assign out = 32'h4;
         |CHECK:      endmodule
         |""".stripMargin
    )

  }

}

```

:::note

FileCheck has _a lot_ of useful features that are not shown in this example.

`CHECK-SAME` allows for checking a match on the same line.  `CHECK-NOT` ensures
that a match does _not_ happen.  `CHECK-COUNT-<n>` will check for `n`
repetitions of a match.  `CHECK-DAG` will allow for a series of matches to occur
in any order.

Most powerfully, FileCheck allows for inline regular expression and saving the
results in string substitution blocks which can then be used later.  This is
useful when you care about capturing a name, but do not care about the actual
name.

Please see the FileCheck documentation for more thorough documentation.

:::
