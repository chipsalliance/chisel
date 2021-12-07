![Chisel 3](https://raw.githubusercontent.com/chipsalliance/chisel3/master/docs/src/images/chisel_logo.svg?sanitize=true)

---

## Upcoming Events

### Chisel Dev Meeting
Chisel/FIRRTL development meetings happen every Monday and Tuesday from 1100--1200 PT.

Call-in info and meeting notes are available [here](https://docs.google.com/document/d/1BLP2DYt59DqI-FgFCcjw8Ddl4K-WU0nHmQu0sZ_wAGo/).

### Chisel Community Conference 2021, Shanghai, China.
CCC is an annual gathering of Chisel community enthusiasts and technical exchange workshop. 
This year with the support of the Chisel development community and RISC-V World Conference China 2021 Committee, we have brought together designers and developers with hands-on experience in Chisel from home and abroad to share cutting-edge results and experiences from both the open source community as well as industry.  
English translated recordings version will be updated soon.  
Looking forward to CCC 2022! See you then!

---

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://badges.gitter.im/chipsalliance/chisel3.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![CI](https://github.com/chipsalliance/chisel3/actions/workflows/test.yml/badge.svg)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/chipsalliance/chisel3.svg?label=release)](https://github.com/chipsalliance/chisel3/releases/latest)

[**Chisel**](https://www.chisel-lang.org) is a hardware design language that facilitates **advanced circuit generation and design reuse for both ASIC and FPGA digital logic designs**.
Chisel adds hardware construction primitives to the [Scala](https://www.scala-lang.org) programming language, providing designers with the power of a modern programming language to write complex, parameterizable circuit generators that produce synthesizable Verilog.
This generator methodology enables the creation of re-usable components and libraries, such as the FIFO queue and arbiters in the [Chisel Standard Library](https://www.chisel-lang.org/api/latest/#chisel3.util.package), raising the level of abstraction in design while retaining fine-grained control.

For more information on the benefits of Chisel see: ["What benefits does Chisel offer over classic Hardware Description Languages?"](https://stackoverflow.com/questions/53007782/what-benefits-does-chisel-offer-over-classic-hardware-description-languages)

Chisel is powered by [FIRRTL (Flexible Intermediate Representation for RTL)](https://github.com/chipsalliance/firrtl), a hardware compiler framework that performs optimizations of Chisel-generated circuits and supports custom user-defined circuit transformations.

## What does Chisel code look like?

Consider an FIR filter that implements a convolution operation, as depicted in this block diagram:

<img src="https://raw.githubusercontent.com/chipsalliance/chisel3/master/docs/src/images/fir_filter.svg?sanitize=true" width="512" />

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
import chisel3.stage.{ChiselStage, ChiselGeneratorAnnotation}

(new chisel3.stage.ChiselStage).execute(
  Array("-X", "verilog"),
  Seq(ChiselGeneratorAnnotation(() => new FirFilter(8, Seq(1.U, 1.U, 1.U)))))
```

Alternatively, you may generate some Verilog directly for inspection:
```scala
val verilogString = (new chisel3.stage.ChiselStage).emitVerilog(new FirFilter(8, Seq(0.U, 1.U)))
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

If you insist on setting up your own project from scratch, your project needs to depend on both the chisel3-plugin (Scalac plugin) and the chisel3 library.
For example, in SBT this could be expressed as:
```scala
// build.sbt
scalaVersion := "2.12.13"
addCompilerPlugin("edu.berkeley.cs" % "chisel3-plugin" % "3.4.4" cross CrossVersion.full)
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.4.4"
scalacOptions += "-Xsource:2.11"
// We also recommend using chiseltest for writing unit tests 
libraryDependencies += "edu.berkeley.cs" %% "chiseltest" % "0.3.4" % "test"
```
### Guide For New Contributors
If you are trying to make a contribution to this project, please read [CONTRIBUTING.md](https://github.com/Burnleydev1/chisel3/blob/recent_PR/CONTRIBUTING.md)

### Design Verification

These simulation-based verification tools are available for Chisel:
- [**iotesters**](https://github.com/freechipsproject/chisel-testers), specifically [PeekPokeTester](https://github.com/freechipsproject/chisel-testers/wiki/Using%20the%20PeekPokeTester), provides constructs (`peek`, `poke`, `expect`) similar to a non-synthesizable Verilog testbench.
- [**testers2**](https://github.com/ucb-bar/chisel-testers2) is an in-development replacement for PeekPokeTester, providing the same base constructs but with a streamlined interface and concurrency support with `fork` and `join`.


## Documentation

### Useful Resources

- [**Cheat Sheet**](https://github.com/freechipsproject/chisel-cheatsheet/releases/latest/download/chisel_cheatsheet.pdf), a 2-page reference of the base Chisel syntax and libraries
- [**ScalaDoc**](https://www.chisel-lang.org/api/latest/chisel3/index.html), a listing, description, and examples of the functionality exposed by Chisel
- [**Gitter**](https://gitter.im/freechipsproject/chisel3), where you can ask questions or discuss anything Chisel
- [**Website**](https://www.chisel-lang.org) ([source](https://github.com/freechipsproject/www.chisel-lang.org/))
- [**Scastie (3.5.0-RC1)**](https://scastie.scala-lang.org/KtzZQ3nFTea9KoNh0tRqtg)
- [**asic-world**](http://www.asic-world.com/verilog/veritut.html) If you aren't familiar with verilog, this is a good tutorial.

If you are migrating from Chisel2, see [the migration guide](https://www.chisel-lang.org/chisel3/chisel3-vs-chisel2.html).

### Data Types Overview
These are the base data types for defining circuit components:

![Image](https://raw.githubusercontent.com/chipsalliance/chisel3/master/docs/src/images/type_hierarchy.svg?sanitize=true)

## Contributor Documentation
This section describes how to get started contributing to Chisel itself, including how to test your version locally against other projects that pull in Chisel using [sbt's managed dependencies](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html).

### Useful Resources for Contributors

The [Useful Resources](#useful-resources) for users are also helpful for contributors.

- [**Chisel Breakdown Slides**](https://docs.google.com/presentation/d/114YihixFBPCfUnv1inqAL8UjsiWfcNWdPHX7SeqlRQc), an introductory talk about Chisel's internals

### Compiling and Testing Chisel

You must first install required dependencies to build Chisel locally, please see [the setup instructions](SETUP.md).

Clone and build the Chisel library:

```bash
git clone https://github.com/chipsalliance/chisel3.git
cd chisel3
sbt compile
```

In order to run the following unit tests, you will need several tools on your `PATH`, namely
[verilator](https://www.veripool.org/verilator/),
[yosys](http://www.clifford.at/yosys/),
[espresso](https://github.com/chipsalliance/espresso),
and [z3](https://github.com/Z3Prover/z3).
Check that each is installed on your `PATH` by running `which verilator` and so on.

If the compilation succeeded and the dependencies noted above are installed, you can then run the included unit tests by invoking:

```
sbt test
```

### Running Projects Against Local Chisel

To use the development version of Chisel (`master` branch), you will need to build from source and `publishLocal`.
The repository version can be found in the [build.sbt](build.sbt) file.
As of the time of writing it was:

```
version := "3.4-SNAPSHOT"
```

To publish your version of Chisel to the local Ivy (sbt's dependency manager) repository, run:

```
sbt publishLocal
```

The compiled version gets placed in `~/.ivy2/local/edu.berkeley.cs/`.
If you need to un-publish your local copy of Chisel, remove the directory generated in `~/.ivy2/local/edu.berkeley.cs/`.

In order to have your projects use this version of Chisel, you should update the `libraryDependencies` setting in your project's build.sbt file to:

```
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.4-SNAPSHOT"
```

### Building Chisel with FIRRTL in the same SBT Project

While we recommend using the library dependency approach as described above, it is possible to build Chisel and FIRRTL in a single SBT project.

**Caveats**
* This only works for the "main" configuration; you cannot build the Chisel tests this way because `treadle` is only supported as a library dependency.
* Do not `publishLocal` when building this way. The published artifact will be missing the FIRRTL dependency.

This works by using [sbt-sriracha](http://eed3si9n.com/hot-source-dependencies-using-sbt-sriracha), an SBT plugin for toggling between source and library dependencies.
It provides two JVM system properties that, when set, will tell SBT to include FIRRTL as a source project:
* `sbt.sourcemode` - when set to true, SBT will look for FIRRTL in the workspace
* `sbt.workspace` - sets the root directory of the workspace

Example use:
```bash
# From root of this repo
git clone git@github.com:chipsalliance/firrtl.git
sbt -Dsbt.sourcemode=true -Dsbt.workspace=$PWD
```

This is primarily useful for building projects that themselves want to include Chisel as a source dependency.
As an example, see [Rocket Chip](https://github.com/chipsalliance/rocket-chip)

### Chisel3 Architecture Overview

The Chisel3 compiler consists of these main parts:

- **The frontend**, `chisel3.*`, which is the publicly visible "API" of Chisel
  and what is used in Chisel RTL. These just add data to the...
- **The Builder**, `chisel3.internal.Builder`, which maintains global state
  (like the currently open Module) and contains commands, generating...
- **The intermediate data structures**, `chisel3.firrtl.*`, which are
  syntactically very similar to Firrtl. Once the entire circuit has been
  elaborated, the top-level object (a `Circuit`) is then passed to...
- **The Firrtl emitter**, `chisel3.firrtl.Emitter`, which turns the
  intermediate data structures into a string that can be written out into a
  Firrtl file for further processing.

Also included is:
- **The standard library** of circuit generators, `chisel3.util.*`. These
  contain commonly used interfaces and constructors (like `Decoupled`, which
  wraps a signal with a ready-valid pair) as well as fully parameterizable
  circuit generators (like arbiters and multiplexors).
- **Chisel Stage**, `chisel3.stage.*`, which contains compilation and test
  functions that are invoked in the standard Verilog generation and simulation
  testing infrastructure. These can also be used as part of custom flows.
  
### Chisel Sub-Projects

Chisel consists of 4 Scala projects; each is its own separate compilation unit:

- [`core`](core) is the bulk of the source code of Chisel, depends on `macros`
- [`src/main`](src/main) is the "main" that brings it all together and includes a [`util`](src/main/scala/chisel3/util) library, which depends on `core`
- [`plugin`](plugin) is the compiler plugin, no internal dependencies
- [`macros`](macros) is most of the macros used in Chisel, no internal dependencies

Code that touches lots of APIs that are private to the `chisel3` package should belong in `core`, while code that is pure Chisel should belong in `src/main`.

### Which version should I use?

The chisel eco-system (`chisel3`, `firrtl`, `dsptools`, `firrtl-interpreter`, `treadle`, `diagrammer`) use a form of semantic versioning:
 major versions are identified by two leading numbers, separated by a dot (i.e., `3.2`), minor versions by a single number following the major version, separated by a dot.
 We maintain API compatibility within a major version (i.e., `3.2.12` should be API-compatible with `3.2.0`), but do not guarantee API compatibility between major versions
 (i.e., APIs may change between `3.1.8` and `3.2.0`).
 We may introduce new definitions or add additional parameters to existing definitions in a minor release, but we do our best to maintain compatibility with previous minor releases of a major release - code that worked in `3.2.0` should continue to work un-modified in `3.2.10`.

We encourage chisel users (rather than chisel developers), to use release versions of chisel.
 The chisel web site (and GitHub repository) should indicate the current release version.
 If you encounter an issue with a released version of chisel, please file an issue on GitHub mentioning the chisel version and provide a simple test case (if possible).
 Try to reproduce the issue with the associated latest minor release (to verify that the issue hasn't been addressed).

If you're developing a chisel library (or `chisel` itself), you'll probably want to work closer to the tip of the development trunk.
 By default, the master branches of the chisel repositories are configured to build and publish their version of the code as `Z.Y-SNAPSHOT`.
 We try to publish an updated SNAPSHOT every two weeks.
 There is no guarantee of API compatibility between SNAPSHOT versions, but we publish date-stamped `Z.Y-yyyymmdd-SNAPSHOT` versions which will not change.
 The code in `Z.Y-SNAPSHOT` should match the code in the most recent `Z.Y-yyyymmdd-SNAPSHOT` version, the differences being the chisel library dependencies:
 `Z.Y-SNAPSHOT`s depend on `V.U-SNAPSHOT`s and `Z.Y-yyyymmdd-SNAPSHOT`s will depend on `V.U-yyyymmdd-SNAPSHOT`s.
 **NOTE**: Prior to the `v3.2-20191030-SNAPSHOT` version, we used `Z.Y-mmddyy-SNAPSHOT` to tag and name published SNAPSHOTs.

If you're developing a library (or another chisel tool), you should probably work with date-stamped SNAPSHOTs until your library or tool is ready to be published (to ensure a consistent API).
 Prior to publishing, you should verify your code against generic (no date-stamp) SNAPSHOTs, or locally published clones of the current master branches of chisel dependencies.
