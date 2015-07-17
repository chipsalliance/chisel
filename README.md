# Chisel3
chisel3 is a new FIRRTL based chisel

the current backward incompatiabilities with chisel 2.x are:

```scala
val wire = Bits(width = 15)
```
is

```scala
val wire = Wire(Bits(width = 15))
```

## Chisel3 Infrastructure.

Chisel3 is much more modular than Chisel2. What was once provided by a
monolithic Scala program, is provided by separate components.

Currently, those components are:
 - Chisel3 (Scala)
 - firrtl (Stanza)

and for the C++ simulator
 - filter (Python/C++)
 - flo-llvm (C++)
 - clang

firrtl can generate Verilog output directly, so fewer components are
required for Verilog testing.

### Stanza
In order to build firrtl, you need a (currently patched) copy of
Stanza. (We should add this to the firrtl repo in utils/bin.)

### firrtl
We assume that copies (or links to) firrtl and filter are in
chisel3/bin. flo-llvm and clang should be found in your $PATH.

Follow the instructions on the firrtl repo for building firrtl and put
the resulting binary (utils/bin/firrtl) in chisel3/bin.

### filter
filter is available from the chisel3 repo as a .cpp file. ucbjrl has a
Python version. You could implment it using sed or awk.

### flo-llvm
flo-llvm is Palmer's flo to (.o,.v) converter. It's hosted at:
	 https://github.com/ucb-bar/flo
and
	 https://github.com/palmer-dabbelt/flo-llvm


Installation instructions can be found at:
	 https://wiki.eecs.berkeley.edu/dreamer/Main/DistroSetup

### clang
clang is available for Linux and Mac OS X and usually comes installed
with development tools. You need to ensure that the version you're
using is compatible with flo-llvm (currently, clang/llvm 3.6). There
are instructions on the web for managing multiple versions of
clang/llvm.

Once you have all the components in place, build and publish Chisel3:

```shell
% cd chisel3
% sbt clean publish-local
```

# Repos of Test Code
 - Basic Chisel3 tests https://github.com/ucb-bar/chisel3-tests

bin directory contains shell script wrappers that may need editing for your environment (absolute paths)

 - Tutorials https://github.com/ucb-bar/chisel-tutorial

branch chisel3prep should be buildable with either Chisel2 or Chisel3.

```shell
% make chiselVersion=2.3-SNAPSHOT clean check
```
or
```shell
% make chiselVersion=3.0 clean check
```
