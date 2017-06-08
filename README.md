# Firrtl
[![Build Status](https://travis-ci.org/freechipsproject/firrtl.svg?branch=master)](https://travis-ci.org/freechipsproject/firrtl)
#### Flexible Internal Representation for RTL

 Firrtl is an intermediate representation (IR) for digital circuits designed as a platform for writing circuit-level transformations.
 This repository consists of a collection of transformations (written in Scala) which simplify, verify, transform, or emit their input circuit.

 A Firrtl compiler is constructed by chaining together these transformations, then writing the final circuit to a file.

 For a detailed description of Firrtl's intermediate representation, see the document "Specification of the Firrtl Language" located in [spec/spec.pdf](https://github.com/ucb-bar/firrtl/blob/master/spec/spec.pdf).

 This repository is in ALPHA VERSION, so many things may change in the coming months.

#### Wiki's and Tutorials

Useful information is on our wiki, located here:
* https://github.com/ucb-bar/firrtl/wiki

Some important pages to read, before writing your own transform:
* [Submitting Pull Requests](https://github.com/ucb-bar/firrtl/wiki/submitting-a-pull-request)
* [Understanding Firrtl's IR](https://github.com/ucb-bar/firrtl/wiki/Understanding-Firrtl-Intermediate-Representation)
* [Traversing a Circuit](https://github.com/ucb-bar/firrtl/wiki/traversing-a-circuit)
* [Common Pass Idioms](https://github.com/ucb-bar/firrtl/wiki/Common-Pass-Idioms)

To write a Firrtl transform, please start with the tutorial here: [src/main/scala/tutorial](https://github.com/ucb-bar/firrtl/blob/master/src/main/scala/tutorial).
To run these examples:
```
sbt assembly
./utils/bin/firrtl -td regress -tn rocket --custom-transforms tutorial.lesson1.AnalyzeCircuit
./utils/bin/firrtl -td regress -tn rocket --custom-transforms tutorial.lesson2.AnalyzeCircuit
```

#### Other Tools
* Firrtl syntax highlighting for Vim users: https://github.com/azidar/firrtl-syntax
* Chisel3, an embedded hardware DSL that generates Firrtl: https://github.com/ucb-bar/chisel3
* Firrtl Interpreter: https://github.com/ucb-bar/firrtl-interpreter
* Yosys Verilog-to-Firrtl Front-end: https://github.com/cliffordwolf/yosys

#### Installation Instructions
*Disclaimer*: This project is in alpha, so there is no guarantee anything works. The installation instructions should work for OSX/Linux machines.

##### Prerequisites
 1. If not already installed, install [verilator](http://www.veripool.org/projects/verilator/wiki/Installing) (Requires at least v3.886)
 2. If not already installed, install [sbt](http://www.scala-sbt.org/) (Requires at least v0.13.6)

##### Installation
 1. Clone the repository:
    ```git clone https://github.com/ucb-bar/firrtl.git && cd firrtl```
 1. Compile firrtl: ```sbt compile```
 1. Run tests: ```sbt test```
 1. Build executable (`utils/bin/firrtl`): ```sbt assembly```
    * **Note:** You can add `utils/bin` to your path to call firrtl from other processes
 1. Publish this version locally in order to satisfy other tool chain library dependencies:
```
sbt publish-local
```

##### Useful sbt Tips
 1. Run a single test suite:
 `sbt "testOnly firrtlTests.UnitTests"`
 2. Continually execute a command:
 `sbt ~compile`
 3. Only invoke sbt once:
```
sbt
> compile
> test
```

##### Using Firrtl as a commandline tool
```
utils/bin/firrtl -i regress/rocket.fir -o regress/rocket.v -X verilog // Compiles rocket-chip to Verilog
utils/bin/firrtl --help // Returns usage string
```

