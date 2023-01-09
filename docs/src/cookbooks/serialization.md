---
layout: docs
title:  "Serialization Cookbook"
section: "chisel3"
---

# Serialization Cookbook

* [Why do I need to serialize Modules](#why-do-i-need-to-serialize-modules)
* [How do I serialize Modules with SerializableModuleGenerator](#how-do-i-seerialize-modules-with-serializablemodulegenerator)

## Why do I need to serialize Modules
Chisel provides a very flexible hardware design experience. However, it sometimes becomes too flexible to design a relative big designs, since parameters of module might come from: 1. Global variables; 2. Outer class; 3. Entropies(time, random). It becomes really hard or impossible to describe "how to reproduce this single module?". This forbids doing unit-test for a module generator, and introduces issues in post-synthesis when doing ECO: a change to Module A might lead to change in Module B.
Thus `SerializableModuleGenerator`, `SerializableModule[T <: SerializableModuleParameter]` and `SerializableModuleParameter` are provided to solve these issues.
For any `SerializableModuleGenerator`, Chisel can automatically serialize and de-serialize it by adding these constraints:
1. the `SerializableModule` should not be inner class, since the outer class is a parameter to it;
1. the `SerializableModule` has and only has one parameter with `SerializableModuleParameter` as its type.
1. the Module neither depends on global variables nor uses non-reproducible functions(random, time, etc), and this should be guaranteed by user, since Scala cannot detect it.

It can provide these benefits:
1. user can use `SerializableModuleGenerator(module: class[SerializableModule], parameter: SerializableModuleParameter)` to auto serialize a Module and its parameter.
1. user can nest `SerializableModuleGenerator` in other serializable parameters to represent a relative large parameter.
1. user can elaborate any `SerializableModuleGenerator` into a single module for testing.


## How do I serialize Modules with `SerializableModuleGenerator`
It is pretty simple and illustrated by example below, the `GCD` Module with `width` as its parameter.

```scala mdoc:silent
import chisel3._
import chisel3.experimental.{SerializableModule, SerializableModuleGenerator, SerializableModuleParameter}
import upickle.default._

// provide serialization functions to GCDSerializableModuleParameter
object GCDSerializableModuleParameter {
  implicit def rwP: ReadWriter[GCDSerializableModuleParameter] = macroRW
}

// Parameter
case class GCDSerializableModuleParameter(width: Int) extends SerializableModuleParameter

// Module
class GCDSerializableModule(val parameter: GCDSerializableModuleParameter)
    extends Module
    with SerializableModule[GCDSerializableModuleParameter] {
  val io = IO(new Bundle {
    val a = Input(UInt(parameter.width.W))
    val b = Input(UInt(parameter.width.W))
    val e = Input(Bool())
    val z = Output(UInt(parameter.width.W))
  })
  val x = Reg(UInt(parameter.width.W))
  val y = Reg(UInt(parameter.width.W))
  val z = Reg(UInt(parameter.width.W))
  val e = Reg(Bool())
  when(e) {
    x := io.a
    y := io.b
    z := 0.U
  }
  when(x =/= y) {
    when(x > y) {
      x := x - y
    }.otherwise {
      y := y - x
    }
  }.otherwise {
    z := x
  }
  io.z := z
}
```
using `write` function in `upickle`, it should return a json string:
```scala mdoc
val j = upickle.default.write(
  SerializableModuleGenerator(
    classOf[GCDSerializableModule],
    GCDSerializableModuleParameter(32)
  )
)
```

You can then read from json string and elaborate the Module:
```scala mdoc:compile-only
circt.stage.ChiselStage.emitSystemVerilog(
  upickle.default.read[SerializableModuleGenerator[GCDSerializableModule, GCDSerializableModuleParameter]](
    ujson.read(j)
  ).module()
)
