![Chisel 3](https://raw.githubusercontent.com/freechipsproject/chisel3/master/doc/images/chisel_logo.svg?sanitize=true)

#

[![Join the chat at https://gitter.im/freechipsproject/chisel3](https://badges.gitter.im/freechipsproject/chisel3.svg)](https://gitter.im/freechipsproject/chisel3?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![CircleCI](https://circleci.com/gh/freechipsproject/chisel3/tree/master.svg?style=shield)](https://circleci.com/gh/freechipsproject/chisel3/tree/master)
[![GitHub tag (latest SemVer)](https://img.shields.io/github/tag/freechipsproject/chisel3.svg?label=release)](https://github.com/freechipsproject/chisel3/releases/latest)

Chisel is a new hardware construction language to support advanced hardware design and circuit generation.
The latest iteration of [Chisel](https://chisel.eecs.berkeley.edu/) is Chisel3,
which uses Firrtl as an intermediate hardware representation language.

Chisel3 releases are available as jars on Sonatype/Nexus/Maven and as tagged branches on the [releases tab](https://github.com/freechipsproject/chisel3/releases) of this repository.

Please visit the [Wiki](https://github.com/ucb-bar/chisel3/wiki) for documentation!

The ScalaDoc for Chisel3 is available at the [API tab on the Chisel web site.](https://chisel.eecs.berkeley.edu/api/latest/index.html)

## Overview
The standard Chisel3 compilation pipeline looks like:
- Chisel3 (Scala) to Firrtl (this is your "Chisel RTL").
- [Firrtl](https://github.com/ucb-bar/firrtl) to Verilog (which can then be passed into FPGA or ASIC tools).
- Verilog to C++ for simulation and testing using [Verilator](http://www.veripool.org/wiki/verilator).

## Installation
This will walk you through installing Chisel and its dependencies:
- [sbt](http://www.scala-sbt.org/), which is the preferred Scala build system and what Chisel uses.
- [Firrtl](https://github.com/ucb-bar/firrtl), which compiles Chisel's IR down to Verilog.
  If you're building from a release branch of Chisel3, separate installation of Firrtl is no longer required: the required jar will be automatically downloaded by sbt.
  If you're building chisel3 from the master branch, you'll need to follow the directions on the [Firrtl repository](https://github.com/ucb-bar/firrtl) to publish a local copy of the required jar.
- [Verilator](http://www.veripool.org/wiki/verilator), which compiles Verilog down to C++ for simulation.
  The included unit testing infrastructure uses this.

### (Ubuntu-like) Linux

1. Install Java
   ```
   sudo apt-get install default-jdk
   ```
1. [Install sbt](http://www.scala-sbt.org/release/docs/Installing-sbt-on-Linux.html),
    which isn't available by default in the system package manager:
    ```
    echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
    sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
    sudo apt-get update
    sudo apt-get install sbt
    ```
1. Install Verilator.
    We currently recommend Verilator version 4.006.
    Follow these instructions to compile it from source.

    1. Install prerequisites (if not installed already):
        ```
        sudo apt-get install git make autoconf g++ flex bison
        ```

    2. Clone the Verilator repository:
        ```
        git clone http://git.veripool.org/git/verilator
        ```

    3. In the Verilator repository directory, check out a known good version:
        ```
        git pull
        git checkout verilator_4_006
        ```

    4. In the Verilator repository directory, build and install:
        ```
        unset VERILATOR_ROOT # For bash, unsetenv for csh
        autoconf # Create ./configure script
        ./configure
        make
        sudo make install
        ```

### Arch Linux

```
yaourt -S firrtl-git verilator sbt
```

### Windows

[Download and install sbt for Windows](https://www.scala-sbt.org/download.html).

#### Simulation on Windows

The chisel3 regression tests use Verilator as the simulator, but Verilator does not work well on Windows natively.
However, Verilator works in [WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) or in other Linux-compatible environments like Cygwin.

Alternatively, if you're using [PeekPokeTester](https://github.com/freechipsproject/chisel-testers) or the [Testers2 alpha](https://github.com/ucb-bar/chisel-testers2), you can use [treadle](https://github.com/freechipsproject/treadle) as the simulation engine.
Treadle is a FIRRTL simulator written in Scala, and works on any platform that can run Scala code.
It can simulate any pure Chisel design, but cannot simulate Verilog code and hence will not work on BlackBoxes / ExtModules which do not have corresponding greybox definitions.

There are no issues with generating Verilog from Chisel, which can be pushed to FPGA or ASIC tools.

### Mac OS X

```
brew install sbt verilator
```

## Getting Started
If you are migrating to Chisel3 from Chisel2, please visit
[Chisel3 vs Chisel2](https://github.com/ucb-bar/chisel3/wiki/Chisel3-vs-Chisel2)

### Resources for Learning Chisel
* [Chisel Bootcamp](https://github.com/freechipsproject/chisel-bootcamp), a collection of interactive Jupyter notebooks that teach Chisel
* [Chisel Tutorial](https://github.com/ucb-bar/chisel-tutorial), a collection of exercises utlizing `sbt`

### Data Types Overview
These are the base data types for defining circuit wires (abstract types which
may not be instantiated are greyed out):

![Image](doc/images/type_hierarchy.png?raw=true)

## For Hardware Engineers
This section describes how to get started using Chisel to create a new RTL
design from scratch.

### [Project Setup](https://github.com/ucb-bar/chisel-template)


### RTL
*TODO: toy example*

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
