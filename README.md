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
 - Chisel3 (Scala) to FIRRTL (this is your "Chisel RTL")
 - FIRRTL to Verilog (which then be passed into FPGA or ASIC tools)
 - Optionally, Verilog to C++ (for simulation and testing)

#### Stanza
In order to build firrtl, you need a (currently patched) copy of
Stanza. (We should add this to the firrtl repo in utils/bin.)

### Hello, World

*TODO: quick "Hello, World" tutorial*

## For Developers

### Environment Setup

*TODO: tools needed*

*TODO: running Scala unit tests locally*

*TODO: running circuit regression tests locally*

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
