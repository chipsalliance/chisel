---
layout: docs
title:  "Enumerations"
section: "chisel3"
---

# ChiselEnum

The ChiselEnum type can be used to reduce the chance of error when encoding mux selectors, opcodes, and functional unit operations.
In contrast with `Chisel.util.Enum`, `ChiselEnum` are subclasses of `Data`, which means that they can be used to define fields in `Bundle`s, including in `IO`s.

## Functionality and Examples

```scala mdoc
// Imports used in the following examples
import circt.stage.ChiselStage
import chisel3._
import chisel3.util._
```

```scala mdoc:invisible
// Helper to print stdout from Chisel elab
// May be related to: https://github.com/scalameta/mdoc/issues/517
import java.io._
import _root_.logger.Logger
def grabLog[T](thunk: => T): (String, T) = {
  val baos = new ByteArrayOutputStream()
  val stream = new PrintStream(baos, true, "utf-8")
  val ret = Logger.makeScope(Nil) {
   Logger.setOutput(stream)
   thunk
  }
  (baos.toString, ret)
}
```

Below we see ChiselEnum being used as mux select for a RISC-V core. While wrapping the object in a package is not required, it is highly recommended as it allows for the type to be used in multiple files more easily.

```scala mdoc
// package CPUTypes {
object AluMux1Sel extends ChiselEnum {
  val selectRS1, selectPC = Value
}
// We can see the mapping by printing each Value
AluMux1Sel.all.foreach(println)
```

Here we see a mux using the AluMux1Sel to select between different inputs.

```scala mdoc
import AluMux1Sel._

class AluMux1Bundle extends Bundle {
  val aluMux1Sel = Input(AluMux1Sel())
  val rs1Out     = Input(Bits(32.W))
  val pcOut      = Input(Bits(32.W))
  val aluMux1Out = Output(Bits(32.W))
}

class AluMux1File extends Module {
  val io = IO(new AluMux1Bundle)

  // Default value for aluMux1Out
  io.aluMux1Out := 0.U

  switch (io.aluMux1Sel) {
    is (selectRS1) {
      io.aluMux1Out := io.rs1Out
    }
    is (selectPC) {
      io.aluMux1Out := io.pcOut
    }
  }
}
```

```scala mdoc:verilog
ChiselStage.emitSystemVerilog(new AluMux1File)
```

ChiselEnum also allows for the user to directly set the Values by passing an `UInt` to `Value(...)`
as shown below. Note that the magnitude of each `Value` must be strictly greater than the one before
it.

```scala mdoc
object Opcode extends ChiselEnum {
    val load  = Value(0x03.U) // i "load"  -> 000_0011
    val imm   = Value(0x13.U) // i "imm"   -> 001_0011
    val auipc = Value(0x17.U) // u "auipc" -> 001_0111
    val store = Value(0x23.U) // s "store" -> 010_0011
    val reg   = Value(0x33.U) // r "reg"   -> 011_0011
    val lui   = Value(0x37.U) // u "lui"   -> 011_0111
    val br    = Value(0x63.U) // b "br"    -> 110_0011
    val jalr  = Value(0x67.U) // i "jalr"  -> 110_0111
    val jal   = Value(0x6F.U) // j "jal"   -> 110_1111
}
```

The user can 'jump' to a value and continue incrementing by passing a start point then using a regular Value definition.

```scala mdoc
object BranchFunct3 extends ChiselEnum {
    val beq, bne = Value
    val blt = Value(4.U)
    val bge, bltu, bgeu = Value
}
// We can see the mapping by printing each Value
BranchFunct3.all.foreach(println)
```

## Casting

You can cast an enum to a `UInt` using `.asUInt`:

```scala mdoc
class ToUInt extends RawModule {
  val in = IO(Input(Opcode()))
  val out = IO(Output(UInt()))
  out := in.asUInt
}
```

```scala mdoc:invisible
// Always need to run Chisel to see if there are elaboration errors
ChiselStage.emitSystemVerilog(new ToUInt)
```

You can cast from a `UInt` to an enum by passing the `UInt` to the apply method of the `ChiselEnum` object:

```scala mdoc
class FromUInt extends Module {
  val in = IO(Input(UInt(7.W)))
  val out = IO(Output(Opcode()))
  out := Opcode(in)
}
```

However, if you cast from a `UInt` to an Enum type when there are undefined states in the Enum values
that the `UInt` could hit, you will see a warning like the following:

```scala mdoc:passthrough
val (log, _) = grabLog(ChiselStage.emitCHIRRTL(new FromUInt))
println(s"```\n$log```")
```

(Note that the name of the Enum is ugly as an artifact of our documentation generation flow, it will
be cleaner in normal use).

You can avoid this warning by using the `.safe` factory method which returns the cast Enum in addition
to a `Bool` indicating if the Enum is in a valid state:

```scala mdoc
class SafeFromUInt extends Module {
  val in = IO(Input(UInt(7.W)))
  val out = IO(Output(Opcode()))
  val (value, valid) = Opcode.safe(in)
  assert(valid, "Enum state must be valid, got %d!", in)
  out := value
}
```

Now there will be no warning:

```scala mdoc:passthrough
val (log2, _) = grabLog(ChiselStage.emitCHIRRTL(new SafeFromUInt))
println(s"```\n$log2```")
```

You can also suppress the warning by using `suppressEnumCastWarning`. This is
primarily used for casting from [[UInt]] to a Bundle type that contains an
Enum, where the [[UInt]] is known to be valid for the Bundle type.

```scala mdoc
class MyBundle extends Bundle {
  val addr = UInt(8.W)
  val op = Opcode()
}

class SuppressedFromUInt extends Module {
  val in = IO(Input(UInt(15.W)))
  val out = IO(Output(new MyBundle()))
  suppressEnumCastWarning {
    out := in.asTypeOf(new MyBundle)
  }
}
```

```scala mdoc:invisible
val (log3, _) = grabLog(ChiselStage.emitCHIRRTL(new SuppressedFromUInt))
assert(log3.isEmpty)
```

## Testing

The _Type_ of the enums values is `<ChiselEnum Object>.Type` which can be useful for passing the values
as parameters to a function (or any other time a type annotation is needed).
Calling `.litValue` on an enum value will return the integer value of that object as a
[`BigInt`](https://www.scala-lang.org/api/2.12.13/scala/math/BigInt.html).

```scala mdoc
def expectedSel(sel: AluMux1Sel.Type): Boolean = sel match {
  case AluMux1Sel.selectRS1 => (sel.litValue == 0)
  case AluMux1Sel.selectPC  => (sel.litValue == 1)
  case _                    => false
}
```

The enum value type also defines some convenience methods for working with `ChiselEnum` values. For example, continuing with the RISC-V opcode
example, one could easily create hardware signal that is only asserted on LOAD/STORE operations (when the enum value is equal to `Opcode.load`
or `Opcode.store`) using the `.isOneOf` method:

```scala mdoc
class LoadStoreExample extends Module {
  val io = IO(new Bundle {
    val opcode = Input(Opcode())
    val load_or_store = Output(Bool())
  })
  io.load_or_store := io.opcode.isOneOf(Opcode.load, Opcode.store)
}
```

```scala mdoc:invisible
// Always need to run Chisel to see if there are elaboration errors
ChiselStage.emitSystemVerilog(new LoadStoreExample)
```

Some additional useful methods defined on the `ChiselEnum` object are:

* `.all`: returns the enum values within the enumeration
* `.getWidth`: returns the width of the hardware type

## Workarounds

As of Chisel v3.4.3 (1 July 2020), the width of the values is always inferred.
To work around this, you can add an extra `Value` that forces the width that is desired.
This is shown in the example below, where we add a field `ukn` to force the width to be 3 bits wide:

```scala mdoc
object StoreFunct3 extends ChiselEnum {
    val sb, sh, sw = Value
    val ukn = Value(7.U)
}
// We can see the mapping by printing each Value
StoreFunct3.all.foreach(println)
```

Signed values are not supported so if you want the value signed, you must cast the UInt with `.asSInt`.

## Additional Resources

The ChiselEnum type is much more powerful than stated above. It allows for Sequence, Vec, and Bundle assignments, as well as a `.next` operation to allow for stepping through sequential states and an `.isValid` for checking that a hardware value is a valid `Value`. The source code for the ChiselEnum can be found [here](https://github.com/chipsalliance/chisel3/blob/2a96767097264eade18ff26e1d8bce192383a190/core/src/main/scala/chisel3/StrongEnum.scala) in the class `EnumFactory`. Examples of the ChiselEnum operations can be found [here](https://github.com/chipsalliance/chisel3/blob/dd6871b8b3f2619178c2a333d9d6083805d99e16/src/test/scala/chiselTests/StrongEnum.scala).
