![Chisel 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/chisel_logo.svg?sanitize=true)

#

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://badges.gitter.im/freechipsproject/chisel3.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/freechipsproject/chisel3/tree/master.svg?style=shield)](https://circleci.com/gh/freechipsproject/chisel3/tree/master)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/freechipsproject/chisel3.svg?label=release)](https://github.com/freechipsproject/chisel3/releases/latest)

[Chisel](https://chisel.eecs.berkeley.edu/) is a powerful hardware construction language to support advanced hardware design and circuit generation.
which uses Firrtl as an intermediate hardware representation language. Chisel promotes re-use through parameterization and 
the ability to build libraries.
Better re-use makes hardware engineers more effective.
Chisel is implemented as an embedded Domain Specific Language or DSL, which adds HDL constructs to [Scala](https://www.scala-lang.org/) optimizing it for building circuits.
Chisel takes advantage of object oriented and functional techniques to create complex circuit generators with a minimum of code.
Parameterized generators with strong type and error checking build re-usable stuff fast.

# What does Chisel code look like?

The following example is a parameterized FIR filter.

![FIR FILTER DIAGRAM 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/fir_filter.svg?sanitize=true)

Users of this module can control the coefficients and bitwidth.
This module 
- defines the IO structure
- conenct the output to the following
- zips up the coefficents, with a sequential series of registers that begin with input, into a series of tuples
- multiplies each of these tuples together to create a sequence of products
- sums up the products

```scala
class FIR(bitWidth: Int, coeffs: Seq[UInt]) extends MultiIOModule {
  val (in, out) = (IO(Input(UInt(bitWidth.W))), IO(Output(UInt(bitWidth.W))))

  out := coeffs
          .zip(in +: coeffs.tail.scan(in) { case a => RegNext(a._1) }) // a sequence of connected regs starting at in
          .map { case (c, r) => c * r }                                // multiply each reg by its coefficient
          .reduce(_ +% _)                                              // add up the products
}
```
> The example here is an extension of an example from the bootcamp.
It may look a bit complicated now, but with the bootcamp it will make sense very quickly.
---

# Get Started -- Do the Bootcamp

The Chisel bootcamp is the fastest way to learn Chisel.
The Bootcamp uses Jupyter to get you up to speed right away.
You can even use it online with nothing to install.

## [Click here to go straight to the Bootcamp Now!](https://github.com/freechipsproject/chisel-bootcamp)

---

# Build circuits with Chisel

When you are ready to build and test your own circuits. Chisel3 uses the Sonatype/Nexus/Maven package management systems to seamlessly deliver the environment and extension
libraries. Set up is easy. The chisel-template repo provides a setup and build environment.

## [Take me to the Chisel-Template](https://github.com/freechipsproject/chisel-template)

For more a more complex template that includes rocket-chip and more, visit the [rebar repository](https://github.com/ucb-bar/project-template).

To learn more about installing Chisel3 visit [The wiki installation page](https://github.com/freechipsproject/chisel3/wiki/installation)

---

# Documentation and other Resources

## [Chisel3 Wiki](https://github.com/freechipsproject/chisel3/wiki)

## [Chisel3 API Documentation](https://chisel.eecs.berkeley.edu/api/latest/chisel3/index.html)

## [Chisel3 Cheat Sheet](https://chisel.eecs.berkeley.edu/doc/chisel-cheatsheet3.pdf)

## [Chisel3 Website](https://www.chisel-lang.org)


## More about Chisel
The standard Chisel3 compilation pipeline looks like:
- Chisel3 (Scala) to Firrtl (this is your "Chisel RTL").
- [Firrtl](https://github.com/ucb-bar/firrtl) to Verilog (which can then be passed into FPGA or ASIC tools).
- Verilog to C++ for simulation and testing using [Verilator](http://www.veripool.org/wiki/verilator).


## Migration
If you are migrating to Chisel3 from Chisel2, please visit
[Chisel3 vs Chisel2](https://github.com/ucb-bar/chisel3/wiki/Chisel3-vs-Chisel2)

### Data Types Overview
These are the base data types for defining circuit wires (abstract types which
may not be instantiated are greyed out):

![Image](doc/images/type_hierarchy.png?raw=true)

## For Hardware Engineers
This section describes how to get started using Chisel to create a new RTL
design from scratch.

### [Project Setup](https://github.com/ucb-bar/chisel-template)

### Verification
*The simulation unit testing infrastructure is still a work in progress.*

See `src/test/scala/chiselTests` for example unit tests. Assert.scala is a
pretty bare-bones unittest which also somewhat verifies the testing system
itself.

Unit tests are written with the ScalaTest unit testing framework, optionally
with ScalaCheck generators to sweep the parameter space where desired.

`BasicTester`-based tests run a single circuit in simulation until either the
test finishes or times out after a set amount of cycles. After compilation,
there is no communication between the testdriver and simulation (unlike
Chisel2's Tester, which allowed dynamic peeks and pokes), so all testvectors
must be determined at compile time.

The circuits run must subclass `BasicTester`, which is a Module with the
addition of a `stop` function which finishes the testbench and reports success.
Any `assert`s that trigger (in either the `BasicTester` or a submodule) will
cause the test to fail. `printf`s will forward to the console.

To write a test using the `BasicTester` that integrates with sbt test, create
a class that extends either `ChiselFlatSpec` (BDD-style testing) or
`ChiselPropSpec` (ScalaCheck generators). In the test content, use
```
assert(execute{ new MyTestModule })
```
where `MyTestModule` is your top-level test circuit that extends
`BasicTester`.

*A more Chisel2-like tester may come in the future.*

### Compiling to Simulation
*TODO: commands to compile project to simulation*

*TODO: running testbenches*

## For Chisel Developers
This section describes how to get started developing Chisel itself, including
how to test your version locally against other projects that pull in Chisel
using [sbt's managed dependencies](https://www.scala-sbt.org/1.x/docs/Library-Dependencies.html).

### Compiling and Testing Chisel
In the Chisel repository directory, run:
```
sbt compile
```
to compile the Chisel library. If the compilation succeeded, you can then run
the included unit tests by invoking:
```
sbt test
```

### Running Projects Against Local Chisel
To use the development version of Chisel (`master` branch), you will need to build from source and `publishLocal`.
The repo version can be found in the build.sbt file.
As of the time of writing it was:

    version := "3.2-SNAPSHOT",

To publish your version of Chisel to the local Ivy (sbt's dependency manager)
repository, run:
```
sbt publishLocal
```

*PROTIP*: sbt can automatically run commands on a source change if you prefix
the command with `~`. For example, the above command to publish Chisel locally
becomes `sbt ~publishLocal`.

[sbt's manual](https://www.scala-sbt.org/1.x/docs/Publishing.html#Publishing+Locally)
recommends that you use a `SNAPSHOT` version suffix to ensure that the local
repository is checked for updates. Since the current default is a `SNAPSHOT`,
and the version number is already incremented compared to the currently
published snapshot, you dont need to change version.

The compiled version gets placed in `~/.ivy2/local/`. You can nuke the relevant
subfolder to un-publish your local copy of Chisel.

In order to have your projects use this version of Chisel, you should update
the libraryDependencies setting in your project's build.sbt file to:
```
libraryDependencies += "edu.berkeley.cs" %% "chisel3" % "3.2-SNAPSHOT"
```

The version specifier in libraryDependencies in the project's build.sbt should
match the version string in your local copy of Chisel's build.sbt.

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
  circuit generators (like arbiters and muxes).
- **Driver utilities**, `chisel3.Driver`, which contains compilation and test
  functions that are invoked in the standard Verilog generation and simulation
  testing infrastructure. These can also be used as part of custom flows.
