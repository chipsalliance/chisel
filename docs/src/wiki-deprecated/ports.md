---
layout: docs
title:  "Ports"
section: "chisel3"
---
Ports are used as interfaces to hardware components.  A port is simply
any `Data` object that has directions assigned to its members.

Chisel provides port constructors to allow a direction to be added
(input or output) to an object at construction time. Primitive port
constructors wrap the type of the port in `Input` or `Output`.

An example port declaration is as follows:
```scala
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

Chisel 3.2+ introduces an API `DataMirror.modulePorts` which can be used to inspect the IOs of any Chisel module, including MultiIOModules, RawModules, and BlackBoxes.

Here is an example of how to use this API:

```scala
import chisel3.experimental.DataMirror

class Adder extends MultiIOModule {
  val a = IO(Input(UInt(8.W)))
  val b = IO(Input(UInt(8.W)))
  val c = IO(Output(UInt(8.W)))
  c := a +& b
}

class Test extends MultiIOModule {
  val adder = Module(new Adder)
  // for debug only
  adder.a := DontCare
  adder.b := DontCare

  // Inspect ports of adder
  // Prints something like this
  /**
    * Found port clock: Clock(IO clock in Adder)
    * Found port reset: Bool(IO reset in Adder)
    * Found port a: UInt<8>(IO a in Adder)
    * Found port b: UInt<8>(IO b in Adder)
    * Found port c: UInt<8>(IO c in Adder)
    */
  DataMirror.modulePorts(adder).foreach { case (name, port) => {
    println(s"Found port $name: $port")
  }}
}

chisel3.Driver.execute(Array[String](), () => new Test)
```
