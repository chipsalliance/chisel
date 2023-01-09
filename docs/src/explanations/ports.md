---
layout: docs
title:  "Ports"
section: "chisel3"
---

# Ports

Ports are used as interfaces to hardware components.  A port is simply
any `Data` object that has directions assigned to its members.

Chisel provides port constructors to allow a direction to be added
(input or output) to an object at construction time. Primitive port
constructors wrap the type of the port in `Input` or `Output`.

An example port declaration is as follows:
```scala mdoc:invisible
import chisel3._
```
```scala mdoc
class Decoupled extends Bundle {
  val ready = Output(Bool())
  val data  = Input(UInt(32.W))
  val valid = Input(Bool())
}
```

After defining ```Decoupled```, it becomes a new type that can be
used as needed for module interfaces or for named collections of
wires.

By folding directions into the object declarations, Chisel is able to
provide powerful wiring constructs described later.

## Inspecting Module ports

(Chisel 3.2+)

Chisel 3.2 introduced `DataMirror.modulePorts` which can be used to inspect the IOs of any Chisel module (this includes modules in both `import chisel3._` and `import Chisel._`, as well as BlackBoxes from each package).
Here is an example of how to use this API:

```scala mdoc
import chisel3.reflect.DataMirror
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.ChiselStage

class Adder extends Module {
  val a = IO(Input(UInt(8.W)))
  val b = IO(Input(UInt(8.W)))
  val c = IO(Output(UInt(8.W)))
  c := a +& b
}

class Test extends Module {
  val adder = Module(new Adder)
  // for debug only
  adder.a := DontCare
  adder.b := DontCare

  // Inspect ports of adder
  // See the result below.
   DataMirror.modulePorts(adder).foreach { case (name, port) => {
    println(s"Found port $name: $port")
  }}
}

ChiselStage.emitSystemVerilog(new Test)
```
