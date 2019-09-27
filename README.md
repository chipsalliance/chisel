![Chisel 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/chisel_logo.svg?sanitize=true)

---

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://badges.gitter.im/freechipsproject/chisel3.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/freechipsproject/chisel3/tree/master.svg?style=shield)](https://circleci.com/gh/freechipsproject/chisel3/tree/master)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/freechipsproject/chisel3.svg?label=release)](https://github.com/freechipsproject/chisel3/releases/latest)

[**Chisel**](https://www.chisel-lang.org) is a hardware design language that facilitates **advanced circuit generation and design reuse for both ASIC and FPGA digital logic designs**.
Chisel adds hardware construction primitives to the [Scala](https://www.scala-lang.org) programming language, providing designers with the power of a modern programming language to write complex, parameterizable circuit generators that produce synthesizable Verilog.
This generator methodology enables the creation of re-usable components and libraries, such as the FIFO queue and arbiters in the [Chisel Standard Library](https://www.chisel-lang.org/api/latest/#chisel3.util.package), raising the level of abstraction in design while retaining fine-grained control.

For more information on the benefits of Chisel see: ["What benefits does Chisel offer over classic Hardware Description Languages?"](https://stackoverflow.com/questions/53007782/what-benefits-does-chisel-offer-over-classic-hardware-description-languages)

Chisel is powered by [FIRRTL (Flexible Intermediate Representation for RTL)](https://github.com/freechipsproject/firrtl), a hardware compiler framework that performs optimizations of Chisel-generated circuits and supports custom user-defined circuit transformations.

## What does Chisel code look like?

Consider an FIR filter that implements a convolution operation, as depicted in this block diagram:

<img src="https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/fir_filter.svg?sanitize=true" width="512" />

While Chisel provides similar base primitives as synthesizable Verilog, and *could* be used as such:

```scala
// 3-point moving average implemented in the style of a FIR filter
class MovingAverage3(bitWidth: Int) extends Module {
  val io = IO(new Bundle {
    val in = Input(UInt(bitWidth.W))
    val out = Output(UInt(bitWidth.W))
  })

  val z1 = RegNext(io.in)
  val z2 = RegNext(z1)

  io.out := (io.in * 1.U) + (z1 * 1.U) + (z2 * 1.U)
}
```

the power of Chisel comes from the ability to create generators, such as n FIR filter that is defined by the list of coefficients:
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
val movingAverage3Filter = FirFilter(8.W, Seq(1.U, 1.U, 1.U))  // same 3-point moving average filter as before
val delayFilter = FirFilter(8.W, Seq(0.U, 1.U))  // 1-cycle delay as a FIR filter
val triangleFilter = FirFilter(8.W, Seq(1.U, 2.U, 3.U, 2.U, 1.U))  // 5-point FIR filter with a triangle impulse response
```


## Getting Started

### Bootcamp Interactive Tutorial
The [**online Chisel Bootcamp**](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master) is the recommended way to get started with and learn Chisel.
**No setup is required** (it runs in the browser), nor does it assume any prior knowledge of Scala.

### Build Your Own Chisel Projects

See [the setup instructions](https://github.com/freechipsproject/chisel3/blob/master/SETUP.md) for how to set up your environment to run Chisel locally.

When you're ready to build your own circuits in Chisel, **we recommend starting from the [Chisel Template](https://github.com/freechipsproject/chisel-template) repository**, which provides a pre-configured project, example design, and testbench. Follow the [chisel-template readme](https://github.com/freechipsproject/chisel-template) to get started.

If you insist on setting up your own project, the magic SBT lines are:
```scala
resolvers ++= Seq(
  Resolver.sonatypeRepo("snapshots"),
  Resolver.sonatypeRepo("releases")
)
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.2-SNAPSHOT"
libraryDependencies += "edu.berkeley.cs" %% "chisel-testers2" % "0.1-SNAPSHOT"
```

### Design Verification

These simulation-based verification tools are available for Chisel:
- [**iotesters**](https://github.com/freechipsproject/chisel-testers), specifically [PeekPokeTester](https://github.com/freechipsproject/chisel-testers/wiki/Using%20the%20PeekPokeTester), provides constructs (`peek`, `poke`, `expect`) similar to a non-synthesizable Verilog testbench.
- [**testers2**](https://github.com/ucb-bar/chisel-testers2) is an in-development replacement for PeekPokeTester, providing the same base constructs but with a streamlined interface and concurrency support with `fork` and `join`.


## Documentation

### Useful Resources

- [**Cheat Sheet**](https://github.com/freechipsproject/chisel-cheatsheet/releases/latest/download/chisel_cheatsheet.pdf), a 2-page reference of the base Chisel syntax and libraries
- [**Wiki**](https://github.com/freechipsproject/chisel3/wiki), which contains various feature-specific tutorials and frequently-asked questions.
- [**ScalaDoc**](https://www.chisel-lang.org/api/latest/chisel3/index.html), a listing, description, and examples of the functionality exposed by Chisel
- [**Gitter**](https://gitter.im/freechipsproject/chisel3), where you can ask questions or discuss anything Chisel
- [**Website**](https://www.chisel-lang.org)

If you are migrating from Chisel2, see [the migration guide](https://www.chisel-lang.org/chisel3/chisel3-vs-chisel2.html).

### Data Types Overview
These are the base data types for defining circuit components:

![Image](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/type_hierarchy.svg?sanitize=true)

## Developer Documentation
This section describes how to get started developing Chisel itself, including how to test your version locally against other projects that pull in Chisel using [sbt's managed dependencies](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html).

### Compiling and Testing Chisel

In the chisel3 repository directory compile the Chisel library:

```
sbt compile
```

If the compilation succeeded, you can then run the included unit tests by invoking:

```
sbt test
```

### Running Projects Against Local Chisel

To use the development version of Chisel (`master` branch), you will need to build from source and `publishLocal`.
The repository version can be found in the build.sbt file.
As of the time of writing it was:

```
version := "3.2-SNAPSHOT"
```

To publish your version of Chisel to the local Ivy (sbt's dependency manager) repository, run:

```
sbt publishLocal
```

The compiled version gets placed in `~/.ivy2/local/edu.berkeley.cs/`.
If you need to un-publish your local copy of Chisel, remove the directory generated in `~/.ivy2/local/edu.berkeley.cs/`.

In order to have your projects use this version of Chisel, you should update the `libraryDependencies` setting in your project's build.sbt file to:

```
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.2-SNAPSHOT"
```

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
- **Driver utilities**, `chisel3.Driver`, which contains compilation and test
  functions that are invoked in the standard Verilog generation and simulation
  testing infrastructure. These can also be used as part of custom flows.
