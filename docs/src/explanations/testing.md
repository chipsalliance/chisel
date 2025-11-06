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
provide stimulus to Chisel modules being tested.

:::info

ChiselSim requires the installation of a compatible simulator tool, like
Verilator or VCS.

:::

To use ChiselSim, mix-in one of the following two traits into a class:

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

For more information see the [Chisel API
documentation](https://www.chisel-lang.org/api) for
`chisel3.simulator.SimulatorAPI`

### Stimulus

Simulation APIs take user provided stimulus and apply it to the
design-under-test (DUT).  There are two mechanisms provided for applying
stimulus: (1) Peek/Poke APIs and (2) reusable stimulus patterns.  The former
provide simple, freeform ways to apply simple directed stimulus.  The latter
provide common stimulus applicable to a wide range of modules.

#### Peek/Poke APIs

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

#### Reusable Stimulus Patterns

While the Peek/Poke APIs are useful for freeform tests, there are a number of
common stimulus patterns that are frequently applied during testing.  E.g.,
bringing a module out of reset or running a simulation until it finishes.  These
patterns are provided in the `chisel3.simulator.stimulus` package.  Currently,
the following stimuli are available:

- `ResetProcedure` will reset a module in a predictable fashion.  This provides
  sufficient spacing for initial blocks to execute at time zero, register/memory
  randomization to happen after that, and reset to assert for a parametric
  number of cycles. (This is the same stimulus used by the `simulate` API.)
- `RunUntilFinished` runs the module for a user-provided number of cycles
  expecting that the simulation will finish cleanly (via `chisel3.stop`) or
  error (via a Chisel assertion).  If the unit runs for the number of cycles
  without asserting or finishing, a simulation assertion is thrown.
- `RunUntilSuccess` runs the module for a user-provided number of cycles
  expecting that the module will assert a success port (indicating success) or
  error (via a Chisel assertion).  The success port must be provided to the
  stimulus as a parameter.

These stimuli are intended to be used via their factory methods.  Most stimuli
provide different factories for different module types.  E.g., the
`ResetProcedure` factory has two methods: `any` which will generate stimulus for
_any_ Chisel module and `module` which can only generate stimulus for subtypes
of `Module`.  The reason for this split is that this specific stimulus needs to
know what the clock and reset ports are in order to apply reset stimulus to
them.  Chisel `Module`s have known clock and reset ports allowing the `module`
stimulus to have just one parameter---the number of cycles to apply the reset
for.  However, a Chisel `RawModule` does not have known clock and reset ports
and user needs to provide more parameters to the factory---the number of reset
cycles _and_ functions to get the clock and reset ports.

For more information see the [Chisel API
documentation](https://www.chisel-lang.org/api) for
`chisel3.simulator.stimulus`.

### Example

The example below shows a basic usage of ChiselSim inside ScalaTest.  This shows
a single test suite, `ChiselSimExample`.  To gain access to ChiselSim methods,
the `ChiselSim` trait is mixed in.  A [testing
style](https://www.scalatest.org/user_guide/selecting_a_style), `AnyFunSpec`, is
also chosen.

In the test, module `Foo` is tested using custom stimulus.  Module `Bar` is
tested using reusable `RunUntilFinished` stimulus.  Module `Baz` is tested using
reusable `RunUntilSuccess` stimulus.  All tests, as written, will pass.

```scala mdoc:silent:reset
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.{RunUntilFinished, RunUntilSuccess}
import chisel3.util.Counter
import org.scalatest.funspec.AnyFunSpec

class ChiselSimExample extends AnyFunSpec with ChiselSim {

  class Foo extends Module {
    val a, b = IO(Input(UInt(8.W)))
    val c = IO(Output(chiselTypeOf(a)))

    private val r = Reg(chiselTypeOf(a))

    r :<= a +% b
    c :<= r
  }

  describe("Baz") {

    it("adds two numbers") {

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

  }

  class Bar extends Module {

    val (_, done) = Counter(true.B, 10)

    when (done) {
      stop()
    }

  }

  describe("Bar") {

    it("terminates cleanly before 11 cycles have elapsed") {

      simulate(new Bar)(RunUntilFinished(11))

    }

  }

  class Baz extends Module {

    val success = IO(Output(Bool()))

    val (_, done) = Counter(true.B, 20)

    success :<= done

  }

  describe("Baz") {

    it("asserts success before 21 cycles have elapsed") {

      simulate(new Baz)(RunUntilSuccess(21, _.success))

    }

  }

}

```

### Scalatest Support

ChiselSim provides a number of features that synergize with Scalatest to improve
the testing experience.

#### Directory Naming

When using ChiselSim in a Scalatest environment, by default a testing directory
structure will be created that matches the Scalatest test "scopes" that are
provided.  Practically, this results in your tests being organized based on how
you organized them in Scalatest.

The root of the testing directory is, by default, `build/chiselsim/`.  You may
change this by overriding the `buildDir` method.

Under the testing directory, you will get one directory for each test suite.  Underneath that, you will get a directory for each test "scope".  E.g., for the test shown in the example above, this will produce the following directory structure:

```
build/chiselsim
└── ChiselSimExample
    ├── Foo
    │   └── adds-two-numbers
    ├── Bar
    │   └── terminates-cleanly-before-11-cycles-have-elapsed
    └── Baz
        └── asserts-success-before-21-cycles-have-elapsed
```

#### Command Line Arguments

Scalatest has support for passing command line arguments to Scalatest using its
`ConfigMap` feature.  ChiselSim wraps this with an improved API for adding
command line arguments to tests, displaying help text, and checking that only
legal arguments are passed.

By default, several command line options are already available for ChiselSim
tests using Scalatest.  You can see these by passing the `-Dhelp=1` argument to
Scalatest.  E.g., this is the help text for the tests shown in the example above:

```
Usage: <ScalaTest> [-D<name>=<value>...]

This ChiselSim ScalaTest test supports passing command line arguments via
ScalaTest's "config map" feature.  To access this, append `-D<name>=<value>` for
a legal option listed below.

Options:

  chiselOpts
      additional options to pass to the Chisel elaboration
  emitVcd
      compile with VCD waveform support and start dumping waves at time zero
  firtoolOpts
      additional options to pass to the firtool compiler
  help
      display this help text
```

The most frequently used of these options is `-DemitVcd=1`.  This will cause
your test to dump a Value Change Dump (VCD) waveform when the test executes.
This is useful if your test fails and you need a waveform to debug why.

There are a number of other command line options that you can optionally mix-in
to your ChiselSim Scalatest test suite that are _not_ automatically available to
ChiselSim.  These are available in the `chisel3.simulator.scalatest.Cli` object:

- `EmitFsdb` adds an `-DemitFsdb=1` option which will cause the simulator, if it
  supports it, to generate an FSDB waveform.
- `EmitVpd` adds an `-DemitVpd=1` option which will cause the simulator, if it
  supports it, to generate an VPD waveform.
- `Scale` adds a `-Dscale=<float>` option.  This provides a way for a user to
  "scale" a test up or down at test-tiem, e.g., to make the test run longer.
  This feature is accessed via the `scaled` method that this trait provides.
- `Simulator` adds a `-Dsimulator=<simulator-name>` argument.  This allows for
  test-time selection of either VCS or verilator as the simulation backend.

:::warning

The `Simulator` command line will automatically disable `Temporal` layers when
running with the Verilator backend.  When running without the `Simulator`
command line and explicitly choosing a simulator, no layers are automatically
disabled, even when running with Verilator.

:::


If the command line option that you want to add is not already available, you
can add a custom option to your test using one of several methods provided in
`chisel3.simulator.scalatest.HasCliOptions`.  The most flexible method is
`addOption`.  This allows you to add an option that may change anything about
the simulation including the Chisel elaboration, FIRRTL compilation, or generic
or backend-specific settings.

More commonly, you just want to add an integer, double, string, or flag-like
options to a test.  For this, simpler option _factories_
(`chisel3.simulator.scalatest.CliOption.{simple, double, int, string, flag}`)
are provided.  After an option has been declared, it can be accessed _within a
test_ using the `getOption` method.

:::warning

The `getOption` method may only be used _inside_ a test.  If used outside a
test, this will cause a runtime exception.

:::

The example below shows how to use the `int` option to set a test-time
configurable seed:

```scala mdoc:reset:silent
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.scalatest.HasCliOptions.CliOption
import chisel3.util.random.LFSR
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec

class ChiselSimExample extends AnyFunSpec with ChiselSim {

  CliOption.int("seed", "the seed to use for the test")

  class Foo(seed: Int) extends Module {
    private val lfsr = LFSR(64, seed = Some(seed))
  }

  describe("Foo") {
    it("generates FIRRTL for a module with a test-time configurable seed") {
      ChiselStage.emitCHIRRTL(new Foo(getOption[Int]("seed").getOrElse(42)))
    }
  }

}
```

:::warning

Be parsimonious with test options.  While they can be useful, they may indicate
an anti-pattern in testing.  If your test is test-time parametric, you are no
longer always testing the same thing.  This can create holes when testing your
Chisel generator if the correct parameters are not tested.

Consider, instead, sweeping over test parameters _within your test_ or by
writing multiple Scalatest tests.

:::

## FileCheck

Sometimes, it is sufficient to directly inspect the result of a generator.  This
testing strategy is particularly relevent if you are trying to create very
specific FIRRTL or SystemVerilog structures or to guarantee exact naming of
specific constructs.

While simple testing can be done with string comparisons, this is often
insufficient as it is necessary to both have a mixture of regular expression
captures and ordering of specific lines.  For this, Chisel provides a native way
to write [FileCheck](https://llvm.org/docs/CommandGuide/FileCheck.html) tests.

:::info

Use of FileCheck tests requires installation of the FileCheck binary.  FileCheck
is typically packaged as part of LLVM.

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

:::note

FileCheck is a tool used extensively in the testing of compilers in the LLVM
ecosystem.  [CIRCT](https://github.com/llvm/circt), the compiler that converts
the FIRRTL that Chisel produces into SystemVerilog, makes heavy use of FileCheck
for its own testing.

:::

When writing FileCheck tests, you will often be using a Chisel API to convert
your Chisel circuit into FIRRTL or SystemVerilog.  Two methods exist to do this
in the `circt.stage.ChiselStage` object:

- `emitCHIRRTL` to generate FIRRTL with a few Chisel extensions
- `emitSystemVerilog` to generate SystemVerilog

Both of these methods take an optional `args` parameter which sets the Chisel
elaboration options.  The latter method has an additional, optional
`firtoolOpts` parameter which controls the `firtool` (FIRRTL compiler) options.

Without any `firtoolOpts` provided to `emitSystemVerilog`, the generated
SystemVerilog may be difficult for you to use FileCheck with due to the default
SystemVerilog lowering, emission, and pretty printing used by `firtool`.  To
make it easier to write your tests, we suggest using the following options:

- `-loweringOptions=emittedLineLength=160` to increase the allowable line
  length.  By default, `firtool` will wrap lines that exceed 80 characters.  You
  may consider using a _very long_ line length (e.g., 8192) to avoid this
  problem altogether.

- `-loweringOptions=disallowLocalVariables` to disable generation of `automatic
  logic` temporaries in always blocks.  This can cause temporaries to spill
  within an always block which may be slightly unexpected.

For more information about `firtool` and its lowering options see the [CIRCT's
Verilog Generation
documentation](https://circt.llvm.org/docs/VerilogGeneration/#controlling-output-style-with-loweringoptions)
or invoke `firtool -help` for a complete list of all supported options.

### Example

The example below shows a FileCheck test that checks that a module has a
specific name and that it has some expected content inside it.  Specifically,
this test is checking that constant propagation happens as expected.  As
written, this test will pass.

```scala mdoc:silent:reset
import chisel3._
import chisel3.testing.scalatest.FileCheck
import circt.stage.ChiselStage
import org.scalatest.funspec.AnyFunSpec

class FileCheckExample extends AnyFunSpec with FileCheck {

  class Baz extends RawModule {

    val out = IO(Output(UInt(32.W)))

    out :<= 1.U(32.W) + 3.U(32.W)

  }

  describe("Foo") {

    it("should simplify the constant computation in its body") {

      ChiselStage.emitSystemVerilog(new Baz).fileCheck()(
        """|CHECK:      module Baz(
           |CHECK-NEXT:   output [31:0] out
           |CHECK:        assign out = 32'h4;
           |CHECK:      endmodule
           |""".stripMargin
        )

    }

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
