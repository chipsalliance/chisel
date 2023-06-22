<a href="https://www.chisel-lang.org">
  <img src="https://raw.githubusercontent.com/chipsalliance/chisel/main/docs/src/images/chisel_logo.svg?sanitize=true" height="60">
</a>
<a href="https://www.chipsalliance.org">
  <img align="right" src="https://raw.githubusercontent.com/chipsalliance/.github/main/profile/images/chips_alliance.svg?sanitize=true" height="60">
</a>

The **Constructing Hardware in a Scala Embedded Language** ([**Chisel**](https://www.chisel-lang.org)) is an open-source hardware description language (HDL) used to describe digital electronics and circuits at the register-transfer level that facilitates **advanced circuit generation and design reuse for both ASIC and FPGA digital logic designs**.

Chisel adds hardware construction primitives to the [Scala](https://www.scala-lang.org) programming language, providing designers with the power of a modern programming language to write complex, parameterizable circuit generators that produce synthesizable Verilog.
This generator methodology enables the creation of re-usable components and libraries, such as the FIFO queue and arbiters in the [Chisel Standard Library](https://www.chisel-lang.org/api/latest/#chisel3.util.package), raising the level of abstraction in design while retaining fine-grained control.

For more information on the benefits of Chisel see: ["What benefits does Chisel offer over classic Hardware Description Languages?"](https://stackoverflow.com/questions/53007782/what-benefits-does-chisel-offer-over-classic-hardware-description-languages)

Chisel is powered by [FIRRTL (Flexible Intermediate Representation for RTL)](https://github.com/chipsalliance/firrtl-spec),
a hardware compiler framework implemented by [LLVM CIRCT](https://github.com/llvm/circt).

Chisel is [permissively licensed](LICENSE) (Apache 2.0) under the guidance of [CHIPS Alliance](https://www.chipsalliance.org).

- [What does Chisel code look like?](#what-does-chisel-code-look-like)
  - [LED blink](#led-blink)
  - [FIR Filter](#fir-filter)
- [Getting Started](#getting-started)
  - [Bootcamp Interactive Tutorial](#bootcamp-interactive-tutorial)
  - [A Textbook on Chisel](#a-textbook-on-chisel)
  - [Build Your Own Chisel Projects](#build-your-own-chisel-projects)
  - [Guide For New Contributors](#guide-for-new-contributors)
  - [Design Verification](#design-verification)
- [Documentation](#documentation)
  - [Useful Resources](#useful-resources)
  - [Chisel Dev Meeting](#chisel-dev-meeting)
  - [Data Types Overview](#data-types-overview)
- [Contributor Documentation](#contributor-documentation)
  - [Useful Resources for Contributors](#useful-resources-for-contributors)
  - [Compiling and Testing Chisel](#compiling-and-testing-chisel)
  - [Running Projects Against Local Chisel](#running-projects-against-local-chisel)
  - [Chisel Architecture Overview](#chisel-architecture-overview)
  - [Chisel Sub-Projects](#chisel-sub-projects)
  - [Which version should I use?](#which-version-should-i-use)
  - [Roadmap](#roadmap)

---

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://matrix.to/img/matrix-badge.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Scaladoc](https://www.javadoc.io/badge/org.chipsalliance/chisel_2.13.svg?color=blue&label=Scaladoc)](https://javadoc.io/doc/org.chipsalliance/chisel_2.13/latest)
![CI](https://github.com/chipsalliance/chisel/actions/workflows/test.yml/badge.svg)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/v/tag/chipsalliance/chisel.svg?include_prereleases&sort=semver)](https://github.com/chipsalliance/chisel/releases/latest)
[![Scala version support](https://index.scala-lang.org/chipsalliance/chisel/chisel/latest-by-scala-version.svg?platform=jvm)](https://index.scala-lang.org/chipsalliance/chisel/chisel)
[![Scala version support (chisel3)](https://index.scala-lang.org/chipsalliance/chisel/chisel3/latest-by-scala-version.svg?platform=jvm)](https://index.scala-lang.org/chipsalliance/chisel/chisel3)
[![Sonatype Snapshots](https://img.shields.io/nexus/s/org.chipsalliance/chisel_2.13?server=https%3A%2F%2Fs01.oss.sonatype.org)](https://s01.oss.sonatype.org/content/repositories/snapshots/org/chipsalliance/chisel_2.13)

## What does Chisel code look like?

### LED blink

```scala
import chisel3._
import chisel3.util.Counter
import circt.stage.ChiselStage

class Blinky(freq: Int, startOn: Boolean = false) extends Module {
  val io = IO(new Bundle {
    val led0 = Output(Bool())
  })
  // Blink LED every second using Chisel built-in util.Counter
  val led = RegInit(startOn.B)
  val (_, counterWrap) = Counter(true.B, freq / 2)
  when(counterWrap) {
    led := ~led
  }
  io.led0 := led
}

object Main extends App {
  // These lines generate the Verilog output
  println(
    ChiselStage.emitSystemVerilog(
      new Blinky(1000),
      firtoolOpts = Array("-disable-all-randomization", "-strip-debug-info")
    )
  )
}
```

Should output the following Verilog:
<!--
Note that you can regenerate the HTML below by using VSCode with extensions:
* Markdown All in One: https://marketplace.visualstudio.com/items?itemName=yzhang.markdown-all-in-one
* Verilog-HDL/SystemVerilog/Bluespec SystemVerilog: https://marketplace.visualstudio.com/items?itemName=mshr-h.VerilogHDL

You then generate the Verilog and place it in a syntax highlighted code block in this file, eg.
```verilog
...
```
You can then run the command: > Markdown All in One: Print current document to HTML
Then you can open the generated HTML and copy-paste
-->
<details>
<summary>Click to expand!</summary>

</code></pre>
<pre><code class="language-verilog"><span class="hljs-comment">// Generated by CIRCT firtool-1.37.0</span>
<span class="hljs-keyword">module</span> Blinky(
  <span class="hljs-keyword">input</span>  clock,
         reset,
  <span class="hljs-keyword">output</span> io_led0
);

  <span class="hljs-keyword">reg</span>       led;
  <span class="hljs-keyword">reg</span> [<span class="hljs-number">8</span>:<span class="hljs-number">0</span>] counterWrap_c_value;
  <span class="hljs-keyword">always</span> @(<span class="hljs-keyword">posedge</span> clock) <span class="hljs-keyword">begin</span>
    <span class="hljs-keyword">if</span> (reset) <span class="hljs-keyword">begin</span>
      led &lt;= <span class="hljs-number">1&#x27;h0</span>;
      counterWrap_c_value &lt;= <span class="hljs-number">9&#x27;h0</span>;
    <span class="hljs-keyword">end</span>
    <span class="hljs-keyword">else</span> <span class="hljs-keyword">begin</span>
      <span class="hljs-keyword">automatic</span> <span class="hljs-keyword">logic</span> counterWrap = counterWrap_c_value == <span class="hljs-number">9&#x27;h1F3</span>;
      led &lt;= counterWrap ^ led;
      <span class="hljs-keyword">if</span> (counterWrap)
        counterWrap_c_value &lt;= <span class="hljs-number">9&#x27;h0</span>;
      <span class="hljs-keyword">else</span>
        counterWrap_c_value &lt;= counterWrap_c_value + <span class="hljs-number">9&#x27;h1</span>;
    <span class="hljs-keyword">end</span>
  <span class="hljs-keyword">end</span> <span class="hljs-comment">// always @(posedge)</span>
  <span class="hljs-keyword">assign</span> io_led0 = led;
<span class="hljs-keyword">endmodule</span>
</code></pre>

</details>

### FIR Filter

Consider an FIR filter that implements a convolution operation, as depicted in this block diagram:

<img src="https://raw.githubusercontent.com/chipsalliance/chisel/master/docs/src/images/fir_filter.svg?sanitize=true" width="512" />

While Chisel provides similar base primitives as synthesizable Verilog, and *could* be used as such:

```scala
// 3-point moving sum implemented in the style of a FIR filter
class MovingSum3(bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(bitWidth.W))
    val out = Output(UInt(bitWidth.W))
  })

  val z1 = RegNext(io.in)
  val z2 = RegNext(z1)

  io.out := (io.in * 1.U) + (z1 * 1.U) + (z2 * 1.U)
}
```

the power of Chisel comes from the ability to create generators, such as an FIR filter that is defined by the list of coefficients:

```scala
// Generalized FIR filter parameterized by the convolution coefficients
class FirFilter(bitWidth: Int, coeffs: Seq[UInt]) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(bitWidth.W))
    val out = Output(UInt(bitWidth.W))
  })
  // Create the serial-in, parallel-out shift register
  val zs = Reg(Vec(coeffs.length, UInt(bitWidth.W)))
  zs(0) := io.in
  for (i <- 1 until coeffs.length) {
    zs(i) := zs(i-1)
  }

  // Do the multiplies
  val products = VecInit.tabulate(coeffs.length)(i => zs(i) * coeffs(i))

  // Sum up the products
  io.out := products.reduce(_ + _)
}
```

and use and re-use them across designs:

```scala
val movingSum3Filter = Module(new FirFilter(8, Seq(1.U, 1.U, 1.U)))  // same 3-point moving sum filter as before
val delayFilter = Module(new FirFilter(8, Seq(0.U, 1.U)))  // 1-cycle delay as a FIR filter
val triangleFilter = Module(new FirFilter(8, Seq(1.U, 2.U, 3.U, 2.U, 1.U)))  // 5-point FIR filter with a triangle impulse response
```

The above can be converted to Verilog using `ChiselStage`:

```scala
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{ChiselStage, FirtoolOption}

(new ChiselStage).execute(
  Array("--target", "systemverilog"),
  Seq(ChiselGeneratorAnnotation(() => new FirFilter(8, Seq(1.U, 1.U, 1.U))),
    FirtoolOption("--disable-all-randomization"))
)
```

Alternatively, you may generate some Verilog directly for inspection:

```scala
val verilogString = chisel3.getVerilogString(new FirFilter(8, Seq(0.U, 1.U)))
println(verilogString)
```

## Getting Started

### Bootcamp Interactive Tutorial

The [**online Chisel Bootcamp**](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master) is the recommended way to get started with and learn Chisel.
**No setup is required** (it runs in the browser), nor does it assume any prior knowledge of Scala.

The [**classic Chisel tutorial**](https://github.com/ucb-bar/chisel-tutorial) contains small exercises and runs on your computer.

### A Textbook on Chisel

If you like a textbook to learn Chisel and also a bit of digital design in general, you may be interested in reading [**Digital Design with Chisel**](http://www.imm.dtu.dk/~masca/chisel-book.html). It is available in English, Chinese, Japanese, and Vietnamese.

### Build Your Own Chisel Projects

See [the setup instructions](SETUP.md) for how to set up your environment to build Chisel locally.

When you're ready to build your own circuits in Chisel, **we recommend starting from the [Chisel Template](https://github.com/freechipsproject/chisel-template) repository**, which provides a pre-configured project, example design, and testbench.
Follow the [chisel-template README](https://github.com/freechipsproject/chisel-template) to get started.

If you insist on setting up your own project from scratch, your project needs to depend on both the chisel-plugin (Scalac plugin) and the chisel library.
For example, in SBT this could be expressed as:
```scala
// build.sbt
scalaVersion := "2.13.10"
val chiselVersion = "5.0.0"
addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion
```

For Chisel prior to v5.0.0, Chisel was published using a different artifact name:
```scala
// build.sbt
scalaVersion := "2.13.10"
addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.6.0" cross CrossVersion.full)
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.6.0"
// We also recommend using chiseltest for writing unit tests
libraryDependencies += "edu.berkeley.cs" %% "chiseltest" % "0.6.0" % "test"
```

### Guide For New Contributors

If you are trying to make a contribution to this project, please read [CONTRIBUTING.md](https://github.com/chipsalliance/chisel/blob/master/CONTRIBUTING.md)

### Design Verification

These simulation-based verification tools are available for Chisel:

* [**svsim**](svsim) is the lightweight testing library for Chisel, included in this repository.
* [**chiseltest (Chisel 5.0 and before)**](https://github.com/ucb-bar/chiseltest) is the batteries-included testing and formal verification library for Chisel-based RTL designs and a replacement for the former PeekPokeTester, providing the same base constructs but with a streamlined interface and concurrency support with `fork` and `join` with internal and Verilator integration for simulations.

## Documentation

### Useful Resources

* [**Cheat Sheet**](https://github.com/freechipsproject/chisel-cheatsheet/releases/latest/download/chisel_cheatsheet.pdf), a 2-page reference of the base Chisel syntax and libraries
* [**ScalaDoc (latest)**](https://www.chisel-lang.org/api/latest/index.html), a listing, description, and examples of the functionality exposed by Chisel, [older versions](https://www.chisel-lang.org/api/) are also available
* [**Gitter**](https://gitter.im/freechipsproject/chisel3), where you can ask questions or discuss anything Chisel
* [**Website (3.6 and earlier)**](https://www.chisel-lang.org) ([source](https://github.com/freechipsproject/www.chisel-lang.org/))
* [**Website (main)**](https://chipsalliance.github.io/chisel) ([source](website)) (Note that this will replace the above after the Chisel 5 release)
* [**Scastie (v5.0.0)**](https://scastie.scala-lang.org/UAQiCxZLR863I3jI1yZ34w) - cannot generate Verilog (firtool does not work in Scastie)
* [**Scastie (v3.6.0)**](https://scastie.scala-lang.org/1XICrlaZQs6ZvxpuKdFdDw) - generates Verilog with legacy Scala FIRRTL Compiler
* [**asic-world**](http://www.asic-world.com/verilog/veritut.html) If you aren't familiar with verilog, this is a good tutorial.

If you are migrating from Chisel2, see [the migration guide](https://www.chisel-lang.org/chisel3/docs/appendix/chisel3-vs-chisel2.html).

### Chisel Dev Meeting

Chisel/FIRRTL development meetings happen every Monday from 9:00-10:00 am PT.

Call-in info and meeting notes are available [here](https://docs.google.com/document/d/1BLP2DYt59DqI-FgFCcjw8Ddl4K-WU0nHmQu0sZ_wAGo/).

### Data Types Overview

These are the base data types for defining circuit components:

![Image](https://raw.githubusercontent.com/chipsalliance/chisel/master/docs/src/images/type_hierarchy.svg?sanitize=true)

## Contributor Documentation

This section describes how to get started contributing to Chisel itself, including how to test your version locally against other projects that pull in Chisel using [sbt's managed dependencies](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html).

### Useful Resources for Contributors

The [Useful Resources](#useful-resources) for users are also helpful for contributors.

* [**Chisel Breakdown Slides**](https://docs.google.com/presentation/d/1gMtABxBEDFbCFXN_-dPyvycNAyFROZKwk-HMcnxfTnU/edit?usp=sharing), an introductory talk about Chisel's internals

### Compiling and Testing Chisel

You must first install required dependencies to build Chisel locally, please see [the setup instructions](SETUP.md).

Clone and build the Chisel library:

```bash
git clone https://github.com/chipsalliance/chisel.git
cd chisel
sbt compile
```

In order to run the following unit tests, you will need several tools on your `PATH`, namely
[firtool](https://github.com/llvm/circt/releases/tag/firtool-1.43.0),
[verilator](https://www.veripool.org/verilator/),
[yosys](https://yosyshq.net/yosys/),
and [espresso](https://github.com/chipsalliance/espresso).
Check that each is installed on your `PATH` by running `which verilator` and so on.

If the compilation succeeded and the dependencies noted above are installed, you can then run the included unit tests by invoking:

```bash
sbt test
```

### Running Projects Against Local Chisel

To use the development version of Chisel (`master` branch), you will need to build from source and publish locally.
The repository version can be found by running `sbt version`.
As of the time of writing it was: `5.0.0-RC1+2-64bbd9ff-SNAPSHOT`.

To publish your version of Chisel to the local Ivy (sbt's dependency manager) repository, run:

```bash
sbt "unipublish / publishLocal"
```

The compiled version gets placed in `~/.ivy2/local/org.chipsalliance/`.
If you need to un-publish your local copy of Chisel, remove the directory generated in `~/.ivy2/local/org.chipsalliance/`.

In order to have your projects use this version of Chisel, you should update the `libraryDependencies` setting in your project's build.sbt file to use the current version, for example:

```scala
val chiselVersion = "5.0.0-RC1+2-64bbd9ff-SNAPSHOT"
addCompilerPlugin("org.chipsalliance" % "chisel-plugin" % chiselVersion cross CrossVersion.full)
libraryDependencies += "org.chipsalliance" %% "chisel" % chiselVersion
```

### Chisel Architecture Overview

The Chisel compiler consists of these main parts:

* **The frontend**, `chisel3.*`, which is the publicly visible "API" of Chisel and what is used in Chisel RTL. These just add data to the...
* **The Builder**, `chisel3.internal.Builder`, which maintains global state (like the currently open Module) and contains commands, generating...
* **The intermediate data structures**, `chisel3.firrtl.*`, which are syntactically very similar to Firrtl. Once the entire circuit has been elaborated, the top-level object (a `Circuit`) is then passed to...
* **The Firrtl emitter**, `chisel3.firrtl.Emitter`, which turns the intermediate data structures into a string that can be written out into a Firrtl file for further processing.

Also included is:

* **The standard library** of circuit generators, `chisel3.util.*`. These contain commonly used interfaces and constructors (like `Decoupled`, which wraps a signal with a ready-valid pair) as well as fully parameterizable circuit generators (like arbiters and multiplexors).
* **Chisel Stage**, `chisel3.stage.*`, which contains compilation and test functions that are invoked in the standard Verilog generation and simulation testing infrastructure. These can also be used as part of custom flows.

### Chisel Sub-Projects

Chisel consists of several Scala projects; each is its own separate compilation unit:

* [`core`](core) is the bulk of the source code of Chisel, depends on `firrtl`, `svsim`, and `macros`
* [`firrtl`](firrtl) is the vestigial remains of the old Scala FIRRTL compiler, much if it will likely be absorbed into `core`
* [`macros`](macros) is most of the macros used in Chisel, no internal dependencies
* [`plugin`](plugin) is the compiler plugin, no internal dependencies
* [`src/main`](src/main) is the "main" that brings it all together and includes a [`util`](src/main/scala/chisel3/util) library, which depends on `core`
* [`svsim`](svsim) is a low-level library for compiling and controlling SystemVerilog simulations, currently targeting Verilator and VCS as backends

Code that touches lots of APIs that are private to the `chisel3` package should belong in `core`, while code that is pure Chisel should belong in `src/main`.

### Which version should I use?

We encourage Chisel users (as opposed to Chisel developers), to use the latest release version of Chisel.
This [chisel-template](https://github.com/freechipsproject/chisel-template) repository is kept up-to-date, depending on the most recent version of Chisel.
The recommended version is also captured near the top of this README, and in the [Github releases](https://github.com/chipsalliance/chisel/releases) section of this repo.
If you encounter an issue with a released version of Chisel, please file an issue on GitHub mentioning the Chisel version and provide a simple test case (if possible).
Try to reproduce the issue with the associated latest minor release (to verify that the issue hasn't been addressed).

For more information on our versioning policy and what versions of the various Chisel ecosystem projects work together, see [Chisel Project Versioning](https://www.chisel-lang.org/chisel3/docs/appendix/versioning.html).

If you're developing a Chisel library (or `chisel3` itself), you'll probably want to work closer to the tip of the development trunk.
By default, the main branch of the chisel repository is configured to build and publish its version of the code as `<version>+<n>-<commit hash>-SNAPSHOT`.
Updated SNAPSHOTs are publised on every push to main.
You are encouraged to do your development against the latest SNAPSHOT, but note that neither API nor ABI compatibility is guaranteed so your code may break at any time.

### Roadmap

See [Roadmap](https://github.com/chipsalliance/chisel3/blob/master/ROADMAP.md).
