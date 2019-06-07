![Chisel 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/chisel_logo.svg?sanitize=true)

#

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://badges.gitter.im/freechipsproject/chisel3.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/freechipsproject/chisel3/tree/master.svg?style=shield)](https://circleci.com/gh/freechipsproject/chisel3/tree/master)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/freechipsproject/chisel3.svg?label=release)](https://github.com/freechipsproject/chisel3/releases/latest)

[**Chisel**](https://chisel-lang.org) is a hardware construction language that facilitates **advanced circuit generation, design reuse, and FPGA/ASIC/technology specialization**.
Chisel is implemented as a [Domain Specific Language (DSL)](https://en.wikipedia.org/wiki/Domain-specific_language) that adds type-safe hardware constructs to the [Scala](https://www.scala-lang.org) programming language.
Using the hardware datatypes of Chisel, the [Chisel Standard Library](https://chisel.eecs.berkeley.edu/api/latest/chisel3/util/index.html), and the multi-paradigm power of the Scala language you can write **complex, highly parameterized circuit generators** that produce synthesizable Verilog.

For more information on the benefits of Chisel see: ["What benefits does Chisel offer over classic Hardware Description Languages?"](https://stackoverflow.com/questions/53007782/what-benefits-does-chisel-offer-over-classic-hardware-description-languages)

# What does Chisel code look like?

The following example is a parameterized FIR filter generator that implements the block diagram below:

![FIR FILTER DIAGRAM 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/fir_filter.svg?sanitize=true)

This module leverages the [functional programming](https://en.wikipedia.org/wiki/Functional_programming) capabilities of Scala to succinctly:

- define the IO structure
- create a shift register of time-delayed inputs
- perform a dot product of time-delayed inputs with coefficients
- connect the result of the dot product to the output

```scala
class FIR(bitWidth: Int, coeffs: Seq[UInt]) extends MultiIOModule {
  val (in, out) = (IO(Input(UInt(bitWidth.W))), IO(Output(UInt(bitWidth.W))))

  out := coeffs
    .zip(in +: coeffs.tail.scan(in) { case a => RegNext(a._1) }) // time-delayed inputs
    .map { case (c, r) => c * r }.reduce(_ +% _)                 // dot product with coeffs
}
```

The above example heavily stresses Scala language features.
Alternatively, Chisel may be used to write Verilog/VHDL-like designs as shown in the following 3-coefficient FIR filter:

```scala
class FIR3(bitWidth: Int) extends Module {
  val io = IO(new Bundle {
      val in = Input(UInt(bitWidth.W))
      val out = Output(UInt(bitWidth.W))
    })

  val b0 = 1.U
  val b1 = 1.U
  val b2 = 1.U

  val z0 = RegNext(io.in)
  val z1 = RegNext(z0)

  io.out := (io.in * b0) + (z0 * b1) + (z1 * b2)
}
```

# Getting Started

## Chisel Bootcamp

The [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp) is the fastest way to learn Chisel.
Here, you can learn about Scala and Chisel and test your knowledge with built-in exercises.

[**Click here to be taken to a hosted instanced of the Chisel Bootcamp and start learning!**](https://mybinder.org/v2/gh/freechipsproject/chisel-bootcamp/master)

## Build Your Own Chisel Projects

After going through the Chisel Bootcamp, you're now ready to build your own hardware.
The [Chisel Template](https://github.com/freechipsproject/chisel-template) repository provides a base setup and build environment for starting a new Chisel project.

## Documentation and Other Resources

We provide a number of informational resources for Chisel:

- [**Chisel3 API Docuementation**](https://chisel.eecs.berkeley.edu/api/latest/chisel3/index.html)
- [**Chisel3 Cheat Sheet**](https://chisel.eecs.berkeley.edu/doc/chisel-cheatsheet3.pdf)
- [**Chisel3 Wiki**](https://github.com/freechipsproject/chisel3/wiki)
- [**Chisel3 Website**](https://www.chisel-lang.org)
- [**Chisel3 Gitter**](https://gitter.im/freechipsproject/chisel3)

# More About Chisel

Chisel forms the first stage of a 3-stage **hardware compiler framework** to generate Verilog.
This is a hardware analog to the [LLVM Compiler Infrastructure Project](https://llvm.org/).
A Chisel design is converted to a circuit intermediate representation (IR) called [FIRRTL (Flexible Intermediate Representation for RTL)](https://github.com/freechipsproject/firrtl).
The second stage, the FIRRTL compiler, optimizes, transforms, and specializes the circuit.
The third stage, a Verilog emitter, converts the FIRRTL to synthesizable Verilog.
This Verilog is then compiled and simulated/tested using [Verilator](http://www.veripool.org/wiki/verilator).
The FIRRTL-produced Verilog can then be passed to an FPGA or ASIC tool chain for deployment or tape-out.

## Migration
If you are migrating to Chisel3 from Chisel2, please visit
[Chisel3 vs Chisel2](https://github.com/ucb-bar/chisel3/wiki/Chisel3-vs-Chisel2).

## Data Types Overview
These are the base data types for defining circuit wires (abstract types which
may not be instantiated are greyed out):

![Image](doc/images/type_hierarchy.png?raw=true)

# Verification

*Testing and verification of Chisel hardware designs happens on the generated Verilog.*

Both the [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp) and [Chisel Template](https://github.com/freechipsproject/chisel-template) provide examples of using Chisel's built in testing libraries.
What follows is an overview of different testing suites for Chisel.

## Basic Tester (chisel3.testers)

This library, packaged with Chisel3, provides basic "hardware testing hardware" style testing using the `BasicTester` class.

## Chisel Testers (chisel3.iotesters)

This library provides "peek, poke, expect" style testing using `PeekPokeTester`.
For more information on Chisel Testers see [freechipsproject/chisel-testers](https://github.com/freechipsproject/chisel-testers).

## Chisel Testers2 (chisel3.tester)

This library, intended to be a replacement for Chisel Testers, provides "peek, poke, expect" style testing in a streamlined interface as well as advanced, necessary testing constructs like "fork" and "join".
For more information on Testers2 see [ucb-bar/chisel-testers2](https://github.com/ucb-bar/chisel-testers2).

# Chisel Development
This section describes how to get started developing Chisel itself, including how to test your version locally against other projects that pull in Chisel using [sbt's managed dependencies](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html).

## Compiling and Testing Chisel

In the chisel3 repository directory compile the Chisel library:

```
sbt compile
```

If the compilation succeeded, you can then run the included unit tests by invoking:

```
sbt test
```

## Running Projects Against Local Chisel

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

## Technical Documentation

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
