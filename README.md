# Chisel3
Chisel3 is a new FIRRTL based chisel.
It is currently in BETA VERSION, so some Chisel features may change in the coming months.

Please visit the [Wiki](https://github.com/ucb-bar/chisel3/wiki) for a more
detailed description.

## Overview
Chisel3 is much more modular than Chisel2, and the compilation pipeline looks
like:
 - Chisel3 (Scala) to FIRRTL (this is your "Chisel RTL").
 - [FIRRTL](https://github.com/ucb-bar/firrtl) to Verilog (which then be passed
 into FPGA or ASIC tools).
 - Verilog to C++ for simulation and testing using
 [Verilator](http://www.veripool.org/wiki/verilator).

## Installation
This will walk you through installing Chisel and its dependencies:
- [sbt](http://www.scala-sbt.org/), which is the preferred Scala build system
  and what Chisel uses.
- [FIRRTL](https://github.com/ucb-bar/firrtl), which compile Chisel's IR down
  to Verilog. A beta version of FIRRTL written in Scala is available.
  - FIRRTL is currently a separate repository but may eventually be made
    available as a standalone program through system package managers and/or
    included in the Chisel source tree.
  - FIRRTL has known issues compiling under JDK 8, which manifests as an
    infinite recursion / stack overflow exception. Instructions for selecting
    JDK 7 are included.
- [Verilator](http://www.veripool.org/wiki/verilator), which compiles Verilog
  down to C++ for simulation. The included unit testing infrastructure uses
  this.

### (Ubuntu-like) Linux

1. [Install sbt](http://www.scala-sbt.org/release/docs/Installing-sbt-on-Linux.html),
  which isn't available by default in the system package manager:

  ```
  echo "deb https://dl.bintray.com/sbt/debian /" | sudo tee -a /etc/apt/sources.list.d/sbt.list
  sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 642AC823
  sudo apt-get update
  sudo apt-get install sbt
  ```
1. Install FIRRTL.
  1. Clone the FIRRTL repository:

    ```
    git clone git@github.com:ucb-bar/firrtl.git
    ```
  1. Build Scala-FIRRTL. In the cloned FIRRTL repository:

    ```
    make build-scala
    ```
    * This compiles FIRRTL into a JAR and creates a wrapper script `firrtl` to
      make the JAR executable. The generated files are in `firrtl/utils/bin`.
    * If this fails with an infinite recursion / stack overflow exception, this
      is a known bug with JDK 8. You can either increase the stack size by
      invoking:

      ```
      JAVA_OPTS=-Xss256m make build-scala
      ```
    * Or, revert to JDK 7:
      1. Install JDK 7 (if not installed already):

        ```
        sudo apt-get install openjdk-7-jdk
        ```
      2. Select JDK 7 as the default JDK:

        ```
        sudo update-alternatives --config java
        ```
  1. Add the FIRRTL executable to your PATH. One way is to add this line to your
    `.bashrc`:

    ```
    export PATH=$PATH:<path-to-your-firrtl-repository>/utils/bin
    ```
1. Install Verilator. As of February 2016, the version of Verilator included by
  in Ubuntu's default package repositories are too out of date, so it must be
  compiled from source.
  1. Install prerequisites (if not installed already):

    ```
    sudo apt-get install git make autoconf g++ flex bison
    ```
  1. Clone the Verilator repository:

    ```
    git clone http://git.veripool.org/git/verilator
    ```
  1. In the Verilator repository directory, check out a known good version:

    ```
    git pull
    git checkout verilator_3_886
    ```
  1. In the Verilator repository directory, build and install:

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

*TODO: write me. If you __really__ want to see this happen, let us know by filing a bug report!*

### Mac OS X

1. Install sbt:

  ```
  brew cask install sbt
  ```
1. Install FIRRTL:
  1. Clone the FIRRTL repository:

    ```
    git clone git@github.com:ucb-bar/firrtl.git
    ```
  1. Build Scala-FIRRTL. In the cloned FIRRTL repository:

    ```
    make build-scala
    ```
    * This compiles FIRRTL into a JAR and creates a wrapper script `firrtl` to
      make the JAR executable. The generated files are in `firrtl/utils/bin`.
    * If this fails with an infinite recursion / stack overflow exception, this
      is a known bug with JDK 8. You can either increase the stack size by
      invoking

      ```
      JAVA_OPTS=-Xss256m make build-scala`
      ```
    * Or, revert to JDK 7:

      ```
      brew install caskroom/versions/java7
      ```
  1. Add the FIRRTL executable to your PATH. One way is to add this line to your
    `.bashrc`:

    ```
    export PATH=$PATH:<path-to-your-firrtl-repository>/utils/bin
    ```
1. Install Verilator:

  ```
  brew install verilator
  ```

## Getting Started
If you are migrating to Chisel3 from Chisel3, please visit
[Chisel3 vs Chisel2](https://github.com/ucb-bar/chisel3/wiki/Chisel3-vs-Chisel2)


### Data Types Overview
These are the base data types for defining circuit wires (abstract types which
may not be instantiated are greyed out):

![Image](doc/images/type_hierarchy.png?raw=true)

### [Chisel Tutorial](https://github.com/ucb-bar/chisel-tutorial)

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
using [sbt's managed dependencies](http://www.scala-sbt.org/0.13/tutorial/Library-Dependencies.html).

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
Chisel3 is still undergoing rapid development and we haven't pusblished a stable version to the Nexus repository.
You will need to build from source and `publish-local`.
The repo version can be found in the build.sbt file.
At last check it was:

      version := "3.0",

To publish your version of Chisel to the local Ivy (sbt's dependency manager)
repository, run:
```
sbt publish-local
```

*PROTIP*: sbt can automatically run commands on a source change if you prefix
the command with `~`. For example, the above command to publish Chisel locally
becomes `sbt ~publish-local`.

[sbt's manual](http://www.scala-sbt.org/0.13/docs/Publishing.html#Publishing+Locally)
recommends that you use a `SNAPSHOT` version suffix to ensure that the local
repository is checked for updates.
Change the version string in build.sbt to:
```
  version := "3.0-SNAPSHOT"
```
and re-execute `sbt publish-local` to accomplish this.

The compiled version gets placed in `~/.ivy2/local/`. You can nuke the relevant
subfolder to un-publish your local copy of Chisel.

In order to have your projects use this version of Chisel, you should update the libraryDependencies setting in your project's build.sbt file to:
```
libraryDependencies += "edu.berkeley.cs" %% "chisel" % "3.0-SNAPSHOT"
```

The version specifier in libraryDependencies in the project's build.sbt should match the version string in your local copy of Chisel's build.sbt.

## Technical Documentation

### Chisel3 Architecture Overview

The Chisel3 compiler consists of these main parts:
 - **The frontend**, `chisel.*`, which is the publicly visible "API" of Chisel
 and what is used in Chisel RTL. These just add data to the...
 - **The Builder**, `chisel.internal.Builder`, which maintains global state
 (like the currently open Module) and contains commands, generating...
 - **The intermediate data structures**, `chisel.firrtl.*`, which are
 syntactically very similar to FIRRTL. Once the entire circuit has been
 elaborated, the top-level object (a `Circuit`) is then passed to...
 - **The FIRRTL emitter**, `chisel.firrtl.Emitter`, which turns the
 intermediate data structures into a string that can be written out into a
 FIRRTL file for further processing.

Also included is:
 - **The standard library** of circuit generators, `chisel.util.*`. These
 contain commonly used interfaces and constructors (like `Decoupled`, which
 wraps a signal with a ready-valid pair) as well as fully parameterizable
 circuit generators (like arbiters and muxes).
 - **Driver utilities**, `chisel.Driver`, which contains compilation and test
 functions that are invoked in the standard Verilog generation and simulation
 testing infrastructure. These can also be used as part of custom flows.
