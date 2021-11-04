---

layout: docs

title:  "Documention of functions"

section: "chisel3"

---

# Further Explanation for the := and  <> Operators

Chisel contains two connection operators, `:=` and `<>`. This document serves for a deeper explanation of the differences of the two and when to use one or the other. The differences are demonstrated with experiments using Scastie examples + DecoupledIO examples.



### Experiment Setup

```scala mdoc
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
Below we can see the resulting Verilog for this example:
```scala modc
ChiselStage.emitVerilog(new Wrapper)
```
## Concept 1: <> is Communicative



This experiment is set up to test for the function of `<>` using the experiment above.

Achieving this involves flipping the RHS and LHS of each arrow and seeing how '<>'  will react.
( Scastie link for the experiment:https://scastie.scala-lang.org/Shorla/LVhlbkFQQnq7X3trHfgZZQ )




```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  // connect Producer to IO
  io.in <> p.io.a
  // connect producer to consumer
  p.io.b <> c.io.a
  // connect consumer to IO
  c.io.b <> io.out
}

class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.b <> io.a
}
```
Below we can see the resulting Verilog for this example:
```scala modc
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion: 
The Verilog remained the same without incurring errors, showing that the `<>` operator is generally communicative.




## Concept 2: := means assign ALL signals from the RHS to the LHS, regardless of their direction.
Using the same experiment code as above, we set to test for the function of ":="
To achieve this, "<>" is being replaced with ":=" in the sample code above and the RHS and LHS are flipped.
(Scastie link to the experiment:https://scastie.scala-lang.org/Shorla/o1ShdaY3RWKf0IIFwwQ1UQ)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  // connect Producer to IO
  io.in := p.io.a
  // connect producer to consumer
  p.io.b := c.io.a
  // connect consumer to IO
  c.io.b := io.out
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.b <> io.a
}
```
Below we can see the resulting Verilog for this example:
```scala modc:crash
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion:
Trying to use ":=" to connect an input will give this error, whether flipped or not.

~~~
chisel3.internal.ChiselException: Connection between sink (DecoupledIO(IO io_in in PipelineStage)) and source (DecoupledIO(IO in unelaborated Wrapper)) failed @.ready: Sink is unwriteable by current module.
	at ... ()
	at Playground$Wrapper.<init>(main.scala:15)
	at Playground$$anonfun$13.apply(main.scala:31)
	at Playground$$anonfun$13.apply(main.scala:31)
	at ... ()
	at ... (Stack trace trimmed to user code only. Rerun with --full-stacktrace to see the full stack trace)
~~~
## Concept 3: Always Use := to assign DontCare to Wires
":=" or "<>" which is best used to assign wires of unknown direction to DontCare? 
We will find that out using the sample codes above.
( Scastie link for the ecperiment:https://scastie.scala-lang.org/Shorla/ZIGsWcylRqKJhZCkKWlSIA/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  //connect Producer to IO
  io.in := DontCare
  p.io.a <> DontCare
  val tmp = Wire(Flipped(DecoupledIO(UInt(8.W))))
  tmp := DontCare
  p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  //connect consumer to IO
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
Below we can see the resulting Verilog for this example:
```scala modc:crash
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion:
When '<>' was used to assign a wire with an unknown direction "tmp" to DontCare, the following error message appeared:
~~~ 
chisel3.internal.ChiselException: Connection between left (DecoupledIO(Wire in Wrapper)) and source (DontCare()) failed @.bits: Locally unclear whether Left or Right (both internal)
	at ... ()
	at Playground$Wrapper.<init>(main.scala:18)
	at Playground$$anonfun$17.apply(main.scala:35)
	at Playground$$anonfun$17.apply(main.scala:35)
	at ... ()
	at ... (Stack trace trimmed to user code only. Rerun with --full-stacktrace to see the full stack trace)
~~~
But when ":=" was used to assign the wire to DontCare, No errors came up. 


##  Concept 4: You can use <> or := to assign DontCare to directioned things (IOs)
':=' or '<>' which is best used to assign directioned things to DontCare? 
We will find that out using the sample codes above.
( Scastie link for the ecperiment:https://scastie.scala-lang.org/Shorla/ZIGsWcylRqKJhZCkKWlSIA/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  //connect Producer to IO
  io.in := DontCare
  p.io.a <> DontCare
  val tmp = Wire(Flipped(DecoupledIO(UInt(8.W))))
  tmp <> DontCare
  p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  //connect consumer to IO
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
Below we can see the resulting Verilog for this example:
```scala modc
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion: 
Both "<>" and ":=" can be used to assign directioned things (IOs) to DontCare as shown in "io.in" and "p.io.a" respectively. This is basically equivalent because in this case both "<>" and ":=" will figure the direction from the LHS.

##  Concept 5: Use <> for intermediate wire connections
The goal is to check if <> can connect two wires using the Experiment code above.
( Scastie link for the ecperiment:https://scastie.scala-lang.org/Shorla/ZIGsWcylRqKJhZCkKWlSIA/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  //connect Producer to IO
  io.in := DontCare
  p.io.a <> DontCare
  val tmp = Wire(Flipped(DecoupledIO(UInt(8.W))))
  tmp := DontCare
  p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  //connect consumer to IO
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
Below we can see the resulting Verilog for this example:
```scala modc:crash
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion:
The code above shows that <> can't connect two wires, this is because Chisel can't figure out which way things flow. If it is used this is the expected error.

~~~
chisel3.internal.ChiselException: Connection between left (DecoupledIO(Wire in Wrapper)) and source (DecoupledIO(Wire in Wrapper)) failed @.bits: Locally unclear whether Left or Right (both internal)
	at ... ()
	at Playground$Wrapper.<init>(main.scala:22)
	at Playground$$anonfun$17.apply(main.scala:40)
	at Playground$$anonfun$17.apply(main.scala:40)
	at ... ()
	at ... (Stack trace trimmed to user code only. Rerun with --full-stacktrace to see the full stack trace)
~~~

## Concept 6: <> only works between things with at least one known flow (An IO or child's IO). 
If there is at least one known flow what will <> do? This will be showed using the experiment code.
( Scastie link for the ecperiment:https://scastie.scala-lang.org/Shorla/gKx9ReLVTTqDTk9vmw5ozg)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage) 
  //connect Producer to IO
    // For this experiment, we add a temporary wire and see if it works...
  //p.io.a <> io.in
  val tmp = Wire(DecoupledIO(UInt(8.W)))
  // connect intermediate wire
  tmp <> io.in
  p.io.a <> tmp
  //p.io.a <> io.in
  // connect producer to consumer
  c.io.a <> p.io.b
  //connect consumer to IO
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
Below we can see the resulting Verilog for this example:
```scala modc
ChiselStage.emitVerilog(new Wrapper)
```
### Conclusion:
The connection above went smoothly with no errors, this goes to show <> will work if there is only one wire in between because the IO and the childIO "fix" the direction.
