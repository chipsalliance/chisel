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

 - Sequential memories:
   ```
   val addr = Reg(UInt())
   val mem = Mem(UInt(width=8), 1024, seqRead = true)
   val dout = when(enable) { mem(addr) }
   ```
   becomes (in Chisel3):
   ```
   val addr = UInt()
   val mem = SeqMem(1024, UInt(width=8))
   val dout = mem.read(addr, enable)
   ```
   Notice the address register is now internal to the SeqMem(), but the data
   will still return on the subsequent cycle. 
  
## Getting Started

### Overview
Chisel3 is much more modular than Chisel2, and the compilation pipeline looks
like:
 - Chisel3 (Scala) to FIRRTL (this is your "Chisel RTL").
 - [FIRRTL](https://github.com/ucb-bar/firrtl) to Verilog (which then be passed
 into FPGA or ASIC tools).
 - Verilog to C++ for simulation and testing using
 [Verilator](http://www.veripool.org/wiki/verilator).

### Installation
To compile down to Verilog for either simulation or synthesis, you will need to 
download and install [FIRRTL](https://github.com/ucb-bar/firrtl). Currently,
FIRRTL is written in Stanza, which means it only runs on Linux or OS X. A
future Scala rewrite is planned which should also allow Windows compatibility.

To compile Verilog down to C++ for simulation (like the included unit testing
infrastructure uses), you will need to have
[Verilator](http://www.veripool.org/wiki/verilator) installed and in your
PATH. Verilator is available via the package manager for some operating systems.

### Data Types Overview
These are the base data types for defining circuit wires (abstract types which
may not be instantiated are greyed out):

![Image](doc/images/type_hierarchy.png?raw=true)

### Chisel Tutorial
*TODO: quick howto for running chisel-tutorial, once chisel-tutorial exists*

## For Hardware Engineers
This section describes how to get started using Chisel to create a new RTL
design from scratch.

### Project Setup
*TODO: recommended sbt style, project structure* 

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
