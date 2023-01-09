---

layout: docs

title:  "Deep Dive into <> and := Connection Operators"

section: "chisel3"

---

# Deep Dive into Connection Operators

Chisel contains two connection operators, `:=` and `<>`. This document provides a deeper explanation of the differences of the two and when to use one or the other. The differences are demonstrated with experiments using Scastie examples which use `DecoupledIO`.



### Experiment Setup

```scala mdoc
// Imports used by the following examples
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage
```

The diagram for the experiment can be viewed [here](https://docs.google.com/document/d/14C918Hdahk2xOGSJJBT-ZVqAx99_hg3JQIq-vaaifQU/edit?usp=sharing).
![Experiment Image](https://raw.githubusercontent.com/chipsalliance/chisel3/master/docs/src/images/connection-operators-experiment.svg?sanitize=true)

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
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
## Concept 1: `<>` is Commutative



This experiment is set up to test for the function of `<>` using the experiment above.

Achieving this involves flipping the RHS and LHS of the `<>` operator and seeing how `<>`  will react.
( Scastie link for the experiment:https://scastie.scala-lang.org/Shorla/LVhlbkFQQnq7X3trHfgZZQ )




```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage)
  // connect producer to I/O
  io.in <> p.io.a
  // connect producer  to consumer
  p.io.b <> c.io.a
  // connect consumer to I/O
  c.io.b <> io.out
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.a <> io.b
}
```
Below we can see the resulting Verilog for this example:
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
### Conclusion:
The Verilog remained the same without incurring errors, showing that the `<>` operator is commutative.




## Concept 2: `:=` means assign ALL LHS signals from the RHS, regardless of the direction on the LHS.
Using the same experiment code as above, we set to test for the function of `:=`
We replace all instances of `<>` with `:=` in the sample code above.
(Scastie link to the experiment: https://scastie.scala-lang.org/Shorla/o1ShdaY3RWKf0IIFwwQ1UQ/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(DecoupledIO(UInt(8.W)))
  val out = DecoupledIO(UInt(8.W))
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage)
  // connect producer to I/O
  p.io.a := io.in
  // connect producer  to consumer
  c.io.a := p.io.b
  // connect consumer to I/O
  io.out := c.io.b
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.a := io.b
}
```
Below we can see the resulting error message for this example:
```scala mdoc:crash
ChiselStage.emitSystemVerilog(new Wrapper)
```
### Conclusion:
The := operator goes field-by-field on the LHS and attempts to connect it to the same-named signal from the RHS. If something on the LHS is actually an Input, or the corresponding signal on the RHS is an Output, you will get an error as shown above.

## Concept 3: Always Use `:=` to assign DontCare to Wires
When assigning `DontCare` to something that is not directioned, should you use `:=` or `<>`?
We will find out using the sample codes below:
( Scastie link for the experiment:https://scastie.scala-lang.org/Shorla/ZIGsWcylRqKJhZCkKWlSIA/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

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
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
### Conclusion:
If `<>` were used to assign the unidrectioned wire `tmp` to DontCare, we would get an error. But in the example above, we used `:=` and no errors occurred.
But when `:=` was used to assign the wire to DontCare, no errors will occur.

Thus, when assigning `DontCare` to a `Wire`, always use `:=`.


##  Concept 4: You can use `<>` or `:=` to assign `DontCare` to directioned things (IOs)
When assigning `DontCare` to something that is directioned, should you use `:=` or `<>`?
We will find out using the sample codes below:
( Scastie link for the experiment:https://scastie.scala-lang.org/Shorla/ZIGsWcylRqKJhZCkKWlSIA/1)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

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
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
### Conclusion:
Both `<>` and `:=` can be used to assign directioned things (IOs) to DontCare as shown in `io.in` and `p.io.a` respectively. This is basically equivalent because in this case both `<>` and `:=` will determine the direction from the LHS.


## Concept 5: `<>`  works between things with at least one known flow (An IO or child's IO).

If there is at least one known flow what will `<>` do? This will be shown using the experiment code below:
( Scastie link for the experiment:https://scastie.scala-lang.org/Shorla/gKx9ReLVTTqDTk9vmw5ozg)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

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
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
### Conclusion:
The connection above went smoothly with no errors, this goes to show `<>` will work as long as there is at least one directioned thing (IO or submodule's IO) to "fix" the direction.


## Concept 6: `<>` and `:=` connect signals by field name.
This experiment creates a MockDecoupledIO which has the same fields by name as a DecoupledIO. Chisel lets us connect it and produces the same verilog, even though MockDecoupledIO and DecoupledIO are different types.
( Scastie link for the experiment:https://scastie.scala-lang.org/Uf4tQquvQYigZAW705NFIQ)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

class MockDecoupledIO extends Bundle {
  val valid = Output(Bool())
  val ready = Input(Bool())
  val bits = Output(UInt(8.W))
}
class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(new MockDecoupledIO())
  val out = new MockDecoupledIO()
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage)
  // connect producer to I/O
  p.io.a <> io.in
  // connect producer  to consumer
  c.io.a <> p.io.b
  // connect consumer to I/O
  io.out <> c.io.b
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.a <> io.b
}
```
Below we can see the resulting Verilog for this example:
```scala mdoc
ChiselStage.emitSystemVerilog(new Wrapper)
```
And here is another experiment, where we remove one of the fields of MockDecoupledIO:
( Scastie link for the experiment:https://scastie.scala-lang.org/ChtkhKCpS9CvJkjjqpdeIA)

```scala mdoc:silent:reset
import chisel3._
import chisel3.util.DecoupledIO
import circt.stage.ChiselStage

class MockDecoupledIO extends Bundle {
  val valid = Output(Bool())
  val ready = Input(Bool())
  //val bits = Output(UInt(8.W))
}
class Wrapper extends Module{
  val io = IO(new Bundle {
  val in = Flipped(new MockDecoupledIO())
  val out = new MockDecoupledIO()
  })
  val p = Module(new PipelineStage)
  val c = Module(new PipelineStage)
  // connect producer to I/O
  p.io.a <> io.in
  // connect producer  to consumer
  c.io.a <> p.io.b
  // connect consumer to I/O
  io.out <> c.io.b
}
class PipelineStage extends Module{
  val io = IO(new Bundle{
    val a = Flipped(DecoupledIO(UInt(8.W)))
    val b = DecoupledIO(UInt(8.W))
  })
  io.a <> io.b
}
```
Below we can see the resulting error for this example:
```scala mdoc:crash
ChiselStage.emitSystemVerilog(new Wrapper)
```
This one fails because there is a field `bits` missing.

### Conclusion:
For `:=`, the Scala types do not need to match but all the signals on the LHS must be provided by the RHS or you will get a Chisel elaboration error. There may be additional signals on the RHS, these will be ignored. For `<>`, the Scala types do not need to match, but all signals must match exactly between LHS and RHS. In both cases, the order of the fields does not matter.
