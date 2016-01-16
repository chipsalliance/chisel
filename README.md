# firrtl
#### Flexible Internal Representation for RTL

 This repository contains the compiler for .fir files.
 It is currently being developed in stanza, with the end-goal to translate into scala for ease of other people to use.
 This decision was made for multiple reasons:
       1.   Previous development had already been done in stanza
       2.   Most of the passes are relatively language independent
       3.   Stanza is easier to develop in than scala bc less-strict type system
       4.   As a favor, its useful to develop in stanza to give Patrick some language tips :)
 The hardest part to port will likely be the parser, but we hope to use an existing scala-based general-purpose parser for the port.

#### Installation Instructions
*Disclaimer*: This project is going through development stages so there is no guarantee anything works.

##### For Linux:
 1. Clone the repository:
 `git clone https://github.com/ucb-bar/firrtl`
 1. Install lit (you need to have pip installed first):
 `pip install lit`
 1. Inflate stanza (this will fail before it builds firrtl):
 `make build`
 1. Remove first empty line in `utils/bin/stanza`, so the first line now becomes `#!/bin/bash`.
 1. Actually build firrtl:
 `make build`
 1. Run `make set-linux`, which links the FileCheck binary to the Linux version.
   * **Note**: This compiled binary may not run on all platforms. You may need to build
     Clang/LLVM from source to extract the compiled FileCheck utility.
 1. Add `firrtl/utils/bin` to your `PATH`, so that the compiled firrtl will be
 available anywhere. This also makes FileCheck available for the tests.
 1. Run tests:
 `make check`
   * **Note**: Stanza eats a 🐣🐣🐣🐣ton of memory and running many instances
     simultaneously (as the build script does) may stall your system due to
     excessive paging. Setting a memory limit seems to make everything behave
     better: `ulimit -v 3096000`
 1. Build and test:
 `make`

##### For Mac:
 1. Clone the repository:
 `git clone https://github.com/ucb-bar/firrtl`
 1. Install lit (you need to have pip installed first):
 `pip install lit`
 1. Build firrtl:
 `make build`
 1. Run `make set-osx`, which links the FileCheck binary to the Mac version.
 1. Run tests:
 `make check`
 1. Build and test:
 `make`

#### Scala implementation
The Scala FIRRTL implementation relies upon sbt 0.13.6. It uses sbt-assembly to create a fat JAR.
Using a bash script and a symbolic link it can be used with the same command-line arguments as the stanza implementation.
Example use:
  1. Build the fat JAR using the makefile: `make build-scala`, or using sbt: `sbt assembly`
    * For development, you can have sbt automatically monitor for changes and recompile: `sbt ~assembly`
  1. Link firrtl to the Scala version: `make set-scala` (can be reverted with `make set-stanza`)
  1. Scala firrtl can be invoked in the same way as Stanza firrtl (and the test
  make targets should continue to work):
  `./utils/bin/firrtl -i <input> -o <output> -X <compiler>`
