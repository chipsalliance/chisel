---
layout: docs
title:  "howitworks"
section: "treadle"
---

## How it works
Treadle ingests a single FIRRTL file runs a few FIRRTL transforms on it
and constructs a symbol table and a program.
The program is nothing more than a complete list of simple assignments topologically sorted so
as single pass of the program will completely update the circuit. 

Treadle has a number of useful capabilities. 
Many of which are not accessible through the standard PeekPoke interface exposed by the chisel unit testesrs.

- All signals and memories and accessible for peeking
- Arbitrary wires can be forced to a particular value
- Verbose modes can show every circuit operation
- A data plugin facility gives developers the ability hooks to monitor every individual assignement statement
  - The VCD output facility is implemented via this plugin architecture
  - Tracing individual assignments and showing their driving inputs is another example
  - An `DataCollector` example that monitors the highest and lowest values recorded on each wire is provided in the tests
- Direct access to one or more clocks and the ability to manually advance wall time is provided.
- Relatively untried capability to allow rollback in time is present
- A Scala based black box facility provides a mechanism for developers to model external verilog functions
  - Treadle cannot otherwise process external verilog black boxes

## Register Handling

## Clock Special Handling

## Memory implementation

## VCD processing