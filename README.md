# firrtl
#### Flexible Internal Representation for RTL

 Firrtl is an intermediate representation (IR) for digital circuits designed as a platform for writing circuit-level transformations.
 This repository consists of a collection of transformations (written in Scala) which simplify, verify, transform, or emit their input circuit.

 A Firrtl compiler is constructed by chaining together these transformations, then writing the final circuit to a file.

 This repository is in ALPHA VERSION, so many things may change in the coming months. 

#### Installation Instructions
*Disclaimer*: This project is in alpha, so there is no guarantee anything works. The installation instructions should work for OSX/Linux machines.

##### Prerequisites
 1. If not already installed, install [verilator](http://www.veripool.org/projects/verilator/wiki/Installing):
 `brew install verilator`
 1. If not already installed, install [sbt](http://www.scala-sbt.org/):
 `brew install sbt`
 * **Note** Requires at least sbt 0.13.6

##### Installation
 1. Clone the repository:
 `git clone https://github.com/ucb-bar/firrtl`
 `cd firrtl`
 1. Compile firrtl:
 `sbt compile`
 1. Run tests:
 `sbt test`
 1. Build executable (utils/bin/firrtl):
 `sbt assembly`
 * **Note** You can add this directory to your path to call firrtl from other processes with th
 1. Run regression:
 `mkdir -p build`
 `./utils/bin/firrtl -i regress/rocket.fir -o build/rocket.v -X verilog

##### Useful sbt Tips
 1. Only invoke sbt once:
 `sbt`
 `> compile`
 `> test`
 1. Run a single test suite:
 `sbt "testOnly firrtlTests.UnitTests"`
 1. Continually execute a command:
 `sbt ~compile`
