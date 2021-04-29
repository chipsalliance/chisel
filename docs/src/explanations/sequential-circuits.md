---
layout: docs
title:  "Sequential Circuits"
section: "chisel3"
---

# Sequential Circuits

```scala mdoc:invisible
import chisel3._
val in = Bool()
```
The simplest form of state element supported by Chisel is a positive edge-triggered register, which can be instantiated as:
```scala mdoc:compile-only
val reg = RegNext(in)
```
This circuit has an output that is a copy of the input signal `in` delayed by one clock cycle. Note that we do not have to specify the type of Reg as it will be automatically inferred from its input when instantiated in this way. In the current version of Chisel, clock and reset are global signals that are implicitly included where needed.

Note that registers which do not specify an initial value will not change value upon toggling the reset signal.

Using registers, we can quickly define a number of useful circuit constructs. For example, a rising-edge detector that takes a boolean signal in and outputs true when the current value is true and the previous value is false is given by:

```scala mdoc:silent
def risingedge(x: Bool) = x && !RegNext(x)
```
Counters are an important sequential circuit. To construct an up-counter that counts up to a maximum value, max, then wraps around back to zero (i.e., modulo max+1), we write:
```scala mdoc:silent
def counter(max: UInt) = {
  val x = RegInit(0.asUInt(max.getWidth.W))
  x := Mux(x === max, 0.U, x + 1.U)
  x
}
```
The counter register is created in the counter function with a reset value of 0 (with width large enough to hold max), to which the register will be initialized when the global reset for the circuit is asserted. The := assignment to x in counter wires an update combinational circuit which increments the counter value unless it hits the max at which point it wraps back to zero. Note that when x appears on the right-hand side of an assignment, its output is referenced, whereas when on the left-hand side, its input is referenced.
Counters can be used to build a number of useful sequential circuits. For example, we can build a pulse generator by outputting true when a counter reaches zero:
```scala mdoc:silent
// Produce pulse every n cycles.
def pulse(n: UInt) = counter(n - 1.U) === 0.U
```
A square-wave generator can then be toggled by the pulse train, toggling between true and false on each pulse:
```scala mdoc:silent
// Flip internal state when input true.
def toggle(p: Bool) = {
  val x = RegInit(false.B)
  x := Mux(p, !x, x)
  x
}
// Square wave of a given period.
def squareWave(period: UInt) = toggle(pulse(period >> 1))
```
