![FIRRTL](https://raw.githubusercontent.com/freechipsproject/firrtl/master/doc/images/firrtl_logo.svg?sanitize=true)

---

[![Join the chat at https://gitter.im/freechipsproject/firrtl](https://badges.gitter.im/freechipsproject/firrtl.svg)](https://gitter.im/freechipsproject/firrtl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
![Build Status](https://github.com/chipsalliance/firrtl/workflows/Continuous%20Integration/badge.svg)
[![Mergify Status][mergify-status]][mergify]

[mergify]: https://mergify.io
[mergify-status]: https://img.shields.io/endpoint.svg?url=https://gh.mergify.io/badges/chipsalliance/firrtl&style=flat

#### Flexible Internal Representation for RTL

 Firrtl is an intermediate representation (IR) for digital circuits designed as a platform for writing circuit-level transformations.
 This repository consists of a collection of transformations (written in Scala) which simplify, verify, transform, or emit their input circuit.

 A Firrtl compiler is constructed by chaining together these transformations, then writing the final circuit to a file.

 For a detailed description of Firrtl's intermediate representation, see the [FIRRTL Language Specification](https://github.com/chipsalliance/firrtl-spec/releases/latest/download/spec.pdf) ([source](https://github.com/chipsalliance/firrtl-spec)).

#### Wiki Pages and Tutorials

Useful information is on our wiki, located here:
* https://github.com/freechipsproject/firrtl/wiki

Some important pages to read, before writing your own transform:
* [Submitting Pull Requests](https://github.com/freechipsproject/firrtl/wiki/Submitting-a-Pull-Request)
* [Understanding Firrtl's IR](https://github.com/freechipsproject/firrtl/wiki/Understanding-Firrtl-Intermediate-Representation)
* [Traversing a Circuit](https://github.com/freechipsproject/firrtl/wiki/traversing-a-circuit)
* [Common Pass Idioms](https://github.com/freechipsproject/firrtl/wiki/Common-Pass-Idioms)

To write a Firrtl transform, please start with the tutorial here: [src/main/scala/tutorial](https://github.com/freechipsproject/firrtl/blob/master/src/main/scala/tutorial).
To run these examples:
```
sbt assembly
./utils/bin/firrtl -td regress -i regress/RocketCore.fir --custom-transforms tutorial.lesson1.AnalyzeCircuit
./utils/bin/firrtl -td regress -i regress/RocketCore.fir --custom-transforms tutorial.lesson2.AnalyzeCircuit
```

#### Other Tools
* Firrtl syntax highlighting for Vim users: https://github.com/azidar/firrtl-syntax
* Firrtl syntax highlighting for Sublime Text 3 users: https://github.com/codelec/highlight-firrtl
* Firrtl syntax highlighting for Atom users: https://atom.io/packages/language-firrtl
* Firrtl syntax highlighting, structure view, navigate to corresponding Chisel code for IntelliJ platform: [install](https://plugins.jetbrains.com/plugin/14183-easysoc-firrtl), [source](https://github.com/easysoc/easysoc-firrtl)
* Firrtl mode for Emacs users: https://github.com/ibm/firrtl-mode
* Chisel3, an embedded hardware DSL that generates Firrtl: https://github.com/freechipsproject/chisel3
* Treadle, a Firrtl Interpreter: https://github.com/freechipsproject/treadle
* Yosys Verilog-to-Firrtl Front-end: https://github.com/cliffordwolf/yosys

#### Installation Instructions
*Disclaimer*: The installation instructions should work for OSX/Linux machines. Other environments may not be tested.

##### Prerequisites
 1. If not already installed, install [verilator](http://www.veripool.org/projects/verilator/wiki/Installing) (Requires at least v3.886)
 1. If not already installed, install [yosys](https://github.com/YosysHQ/yosys) (Requires at least v0.8)
 1. If not already installed, install [sbt](http://www.scala-sbt.org/) (Recommend v1.6.2)

##### Installation
 1. Clone the repository:
    ```git clone https://github.com/freechipsproject/firrtl.git && cd firrtl```
 1. Compile firrtl: ```sbt compile```
 1. Run tests: ```sbt test```
 1. Build executable (`utils/bin/firrtl`): ```sbt assembly```
    * **Note:** You can add `utils/bin` to your path to call firrtl from other processes
 1. Publish this version locally in order to satisfy other tool chain library dependencies:
```
sbt publishLocal
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

##### Use scalafix to remove unused import and deprecated procedure syntax
 1. Remove unused import:
```
sbt "firrtl/scalafix RemoveUnused"
```
 2. Remove deprecated procedure syntax
```
sbt "firrtl/scalafix ProcedureSyntax"
```

##### Using Firrtl as a commandline tool
```
utils/bin/firrtl -i regress/rocket.fir -o regress/rocket.v -X verilog // Compiles rocket-chip to Verilog
utils/bin/firrtl --help // Returns usage string
```

##### Citing Firrtl

If you use Firrtl in a paper, please cite the following ICCAD paper and technical report:
https://ieeexplore.ieee.org/document/8203780
```
@INPROCEEDINGS{8203780,
author={A. Izraelevitz and J. Koenig and P. Li and R. Lin and A. Wang and A. Magyar and D. Kim and C. Schmidt and C. Markley and J. Lawson and J. Bachrach},
booktitle={2017 IEEE/ACM International Conference on Computer-Aided Design (ICCAD)},
title={Reusability is FIRRTL ground: Hardware construction languages, compiler frameworks, and transformations},
year={2017},
volume={},
number={},
pages={209-216},
keywords={field programmable gate arrays;hardware description languages;program compilers;software reusability;hardware development practices;hardware libraries;open-source hardware intermediate representation;hardware compiler transformations;Hardware construction languages;retargetable compilers;software development;virtual Cambrian explosion;hardware compiler frameworks;parameterized libraries;FIRRTL;FPGA mappings;Chisel;Flexible Intermediate Representation for RTL;Reusability;Hardware;Libraries;Hardware design languages;Field programmable gate arrays;Tools;Open source software;RTL;Design;FPGA;ASIC;Hardware;Modeling;Reusability;Hardware Design Language;Hardware Construction Language;Intermediate Representation;Compiler;Transformations;Chisel;FIRRTL},
doi={10.1109/ICCAD.2017.8203780},
ISSN={1558-2434},
month={Nov},}
```

https://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.html
```
@techreport{Li:EECS-2016-9,
    Author = {Li, Patrick S. and Izraelevitz, Adam M. and Bachrach, Jonathan},
    Title = {Specification for the FIRRTL Language},
    Institution = {EECS Department, University of California, Berkeley},
    Year = {2016},
    Month = {Feb},
    URL = {http://www2.eecs.berkeley.edu/Pubs/TechRpts/2016/EECS-2016-9.html},
    Number = {UCB/EECS-2016-9}
}
```
