# firrtl
#### Flexible Internal Representation for RTL

 Firrtl is an intermediate representation (IR) for digital circuits designed as a platform for writing circuit-level transformations.
 This repository consists of a collection of transformations (written in Scala) which simplify, verify, transform, or emit their input circuit.

 A Firrtl compiler is constructed by chaining together these transformations, then writing the final circuit to a file.

 For a detailed description of Firrtl's intermediate representation, see the document "Specification of the Firrtl Language" located in [spec/spec.pdf](https://github.com/ucb-bar/firrtl/blob/master/spec/spec.pdf).

 This repository is in ALPHA VERSION, so many things may change in the coming months.

#### Installation Instructions
*Disclaimer*: This project is in alpha, so there is no guarantee anything works. The installation instructions should work for OSX/Linux machines.

##### Prerequisites
 1. If not already installed, install [verilator](http://www.veripool.org/projects/verilator/wiki/Installing) (Requires at least v3.886)
 2. If not already installed, install [sbt](http://www.scala-sbt.org/) (Requires at least v0.13.6)

##### Installation
 1. Clone the repository:
    ```git clone https://github.com/ucb-bar/firrtl; cd firrtl```
 2. Compile firrtl:```sbt compile```
 3. Run tests: ```sbt test```
 4. Build executable (`utils/bin/firrtl`): ```sbt assembly```
    * **Note:** You can add `utils/bin/firrtl` to your path to call firrtl from other processes
 5. Run regression:
```
mkdir -p build
./utils/bin/firrtl -i regress/rocket.fir -o build/rocket.v -X verilog
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

##### Other Tools
Firrtl syntax highlighting for Vim users: https://github.com/azidar/firrtl-syntax
