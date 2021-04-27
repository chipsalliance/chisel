---
layout: docs
title:  "Functional Module Creation"
section: "chisel3"
---

# Functional Module Creation

Objects in Scala have a pre-existing creation function (method) called `apply`.
When an object is used as value in an expression (which basically means that the constructor was called), this method determines the returned value.
When dealing with hardware modules, one would expect the module output to be representative of the hardware module's functionality.
Therefore, we would sometimes like the module output to be the value returned when using the object as a value in an expression.
Since hardware modules are represented as Scala objects, this can be done by defining the object's `apply` method to return the module's output.
This can be referred to as creating a functional interface for module construction.
If we apply this on the standard mux2 example, we would to return the mux2 output ports when we used mux2 in an expression.
Implementing this requires building a constructor that takes multiplexer inputs as parameters and returns the multiplexer output:

```scala mdoc:silent
import chisel3._

class Mux2 extends Module {
  val io = IO(new Bundle {
    val sel = Input(Bool())
    val in0 = Input(UInt())
    val in1 = Input(UInt())
    val out = Output(UInt())
  })
  io.out := Mux(io.sel, io.in0, io.in1)
}

object Mux2 {
  def apply(sel: UInt, in0: UInt, in1: UInt) = {
    val m = Module(new Mux2)
    m.io.in0 := in0
    m.io.in1 := in1
    m.io.sel := sel
    m.io.out
  }
}
```

As we can see in the code example, we defined the `apply` method to take the Mux2 inputs as the method parameters, and return the Mux2 output as the function's return value.
By defining modules in this way, it is easier to later implement larger and more complex version of this regular module.
For example, we previously implemented Mux4 like this:

```scala mdoc:silent
class Mux4 extends Module {
  val io = IO(new Bundle {
    val in0 = Input(UInt(1.W))
    val in1 = Input(UInt(1.W))
    val in2 = Input(UInt(1.W))
    val in3 = Input(UInt(1.W))
    val sel = Input(UInt(2.W))
    val out = Output(UInt(1.W))
  })
  val m0 = Module(new Mux2)
  m0.io.sel := io.sel(0)
  m0.io.in0 := io.in0
  m0.io.in1 := io.in1

  val m1 = Module(new Mux2)
  m1.io.sel := io.sel(0)
  m1.io.in0 := io.in2
  m1.io.in1 := io.in3

  val m3 = Module(new Mux2)
  m3.io.sel := io.sel(1)
  m3.io.in0 := m0.io.out
  m3.io.in1 := m1.io.out

  io.out := m3.io.out
}
```

However, by using the creation function we redefined for Mux2, we can now use the Mux2 outputs as values of the modules themselves
when writing the Mux4 output expression:

```scala mdoc:invisible:reset
// We need to re-do this to allow us to `reset`
// and then re-define Mux4
import chisel3._

class Mux2 extends Module {
  val io = IO(new Bundle {
    val sel = Input(Bool())
    val in0 = Input(UInt())
    val in1 = Input(UInt())
    val out = Output(UInt())
  })
  io.out := Mux(io.sel, io.in0, io.in1)
}

object Mux2 {
  def apply(sel: UInt, in0: UInt, in1: UInt) = {
    val m = Module(new Mux2)
    m.io.in0 := in0
    m.io.in1 := in1
    m.io.sel := sel
    m.io.out
  }
}
```

```scala mdoc:silent
class Mux4 extends Module {
  val io = IO(new Bundle {
    val in0 = Input(UInt(1.W))
    val in1 = Input(UInt(1.W))
    val in2 = Input(UInt(1.W))
    val in3 = Input(UInt(1.W))
    val sel = Input(UInt(2.W))
    val out = Output(UInt(1.W))
  })
  io.out := Mux2(io.sel(1),
                 Mux2(io.sel(0), io.in0, io.in1),
                 Mux2(io.sel(0), io.in2, io.in3))
}
```

This allows us to write more intuitively readable hardware connection descriptions, which are similar to software expression evaluation.
