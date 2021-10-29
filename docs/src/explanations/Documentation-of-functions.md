---
layout: docs
title:  "Documention of functions"
section: "chisel3"
---
# Documentation of Functions := vs <>

The difference in the function of := and <> has been a major source of debate in lots of conversation.

This document explains the difference by setting up an experiment using Scastie examples + DecoupledIO examples to put the functions to test.

### Experiment

```scala mdoc:silent
// Imports used by the following examples
import chisel3._
import chisel3.util.DecoupledIO
```

The diagram for the experiment can be viewed **[here](https://docs.google.com/document/d/14C918Hdahk2xOGSJJBT-ZVqAx99_hg3JQIq-vaaifQU/edit?usp=sharing)**


```scala mdoc:silent
class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  // connect Producer to IO
  p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  // connect consumer to IO
  io.out <> c.io.b
}

class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.b <> io.a
}
```
Below we can see the resulting verilog for this example:
```scala modc
ChiselStage.emitVerilog(new Wrapper)
```

