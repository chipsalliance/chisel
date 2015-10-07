# Chisel3
Chisel3 is a new FIRRTL based chisel

*TODO: A better description, perhaps lifted off Chisel2's README*

## Chisel2 Migration
For those moving from Chisel2, there were some backwards incompatible changes
and your RTL needs to be modified to work with Chisel3. The required
modifications are: 

 - Wire declaration style:  
   ```
   val wire = Bits(width = 15)
   ```  
   becomes (in Chisel3):  
   ```
   val wire = Wire(Bits(width = 15))
   ```

## Getting Started

### Overview
Chisel3 is much more modular than Chisel2, and the compilation pipeline looks
like:
 - Chisel3 (Scala) to FIRRTL (this is your "Chisel RTL").
 - FIRRTL to Verilog (which then be passed into FPGA or ASIC tools). Repository
 with the compiler and installation instructions are
 [here](https://github.com/ucb-bar/firrtl).
 - Optionally, Verilog to C++ (for simulation and testing).  
 *TODO: Verilator support*

### Data Types Overview
These are the base data types for defining circuit wires: 

![Image](../master/docs/images/type_hierarchy.svg?raw=true)

### Chisel Tutorial
*TODO: quick howto for running chisel-tutorial*

## For Hardware Engineers
This section describes how to get started using Chisel to create a new RTL
design from scratch.

### Project Setup
*TODO: tools needed*

*TODO: recommended sbt style, project structure* 

### RTL and Verification
*TODO: project boilerplate: import statements, main() contents*

*TODO: recommended test structure*

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
sbt *TODO WRITE ME*
``` 

*TODO: circuit test cases*

### Running Projects Against Local Chisel
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

The compiled version gets placed in `~/.ivy2/local/`. You can nuke the relevant
subfolder to un-publish your local copy of Chisel.

## Technical Documentation

### Chisel3 Architecture Overview

The Chisel3 compiler consists of these main parts:
 - **The frontend**, which is the publicly visible "API" of Chisel and what is
 used in Chisel RTL. All these do is build...  
 *TODO: filenames (or package names, once the split is complete*
 - **The intermediate data structures**, which is syntactically very similar
 to FIRRTL. Once the entire circuit has been elaborated, the top-level object
 (a `Circuit`) is then passed to...  
 *TODO: filenames (or package names, once the split is complete*
 - **The FIRRTL emitter**, which turns the intermediate data structures into
 a string that can be written out into a FIRRTL file for further processing.  
 *TODO: filenames (or package names, once the split is complete*
 
Also included is:
 - **The standard library** of circuit generators, currently Utils.scala. These
 contain commonly used interfaces and constructors (like `Decoupled`, which
 wraps a signal with a ready-valid pair) as well as fully parameterizable
 circuit generators (like arbiters and muxes).  
 *TODO: update once standard library gets properly broken you* 
 - *TODO: add details on the testing framework*
 - *TODO: add details on simulators*
