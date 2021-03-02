# ChiselEnum

The ChiselEnum type can be used to reduce the chance of error when encoding mux selectors, opcodes, and functional unit operations. In contrast with`Chisel.util.Enum`, `ChiselEnum` are subclasses of `Data`, which means that they can be used to define fields in `Bundle`s, including in `IO`s.


## Functionality and Examples

```scala mdoc
// Imports used in the following examples
import chisel3._
import chisel3.util._
import chisel3.stage.ChiselStage
import chisel3.experimental.ChiselEnum
```

Below we see ChiselEnum being used as mux select for a RISC-V core. While wrapping the object in a package is not required, it is highly recommended as it allows for the type to be used in multiple files more easily. 

```scala mdoc
// package CPUTypes {
    object AluMux1Sel extends ChiselEnum {
        val selectRS1, selectPC = Value
        /** How the values will be mapped
            "selectRS1" -> 0.U,
            "selectPC"  -> 1.U
        */
    }
// }
```

Here we see a mux using the AluMux1Sel to select between different inputs. 

```scala mdoc
import AluMux1Sel._

class AluMux1Bundle extends Bundle {
        val aluMux1Sel =  Input( AluMux1Sel() )
        val rs1Out     =  Input(Bits(32.W))
        val pcOut      =  Input(Bits(32.W))
        val aluMux1Out = Output(Bits(32.W))
}

class AluMux1File extends Module {
    val io = IO(new AluMux1Bundle)

    // Default value for aluMux1Out
    io.aluMux1Out := 0.U

    switch (io.aluMux1Sel) {
        is (selectRS1) {
            io.aluMux1Out  := io.rs1Out
        }
        is (selectPC) {
            io.aluMux1Out  := io.pcOut
        }
    }
}
```
```scala mdoc:verilog
ChiselStage.emitVerilog(new AluMux1File )
```

ChiselEnum also allows for the user to define variables by passing in the value shown below. Note that the value must be increasing or else 

 > chisel3.internal.ChiselException: Exception thrown when elaborating ChiselGeneratorAnnotation

is thrown during Verilog generation.

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

The user can 'jump' to a value and continue incrementing by passing a start point then using a regular Value assignment. 

```scala mdoc
object BranchFunct3 extends ChiselEnum {
    val beq, bne = Value
    val blt = Value(4.U)
    val bge, bltu, bgeu = Value
    /** How the values will be mapped
        "beq"  -> 0.U,
        "bne"  -> 1.U,
        "blt"  -> 4.U,
        "bge"  -> 5.U,
        "bltu" -> 6.U,
        "bgeu" -> 7.U
    */
}
```

## Testing

When testing your modules, the `.Type` and `.litValue` attributes allow for the the objects to be passed as parameters and for the value to be converted to BigInt type. Note that BigInts cannot be casted to Int with `.asInstanceOf[Int]`, they use their own methods like `toInt`. Please review the [scala.math.BigInt](https://www.scala-lang.org/api/2.12.5/scala/math/BigInt.html) page for more details!

```scala mdoc
def expectedSel(sel: AluMux1Sel.Type): Boolean = sel match {
  case AluMux1Sel.selectRS1 => (sel.litValue == 0)
  case AluMux1Sel.selectPC  => (sel.litValue == 1)
  case _                    => false
}
```

The ChiselEnum type also has methods `.all` and `.getWidth` where `all` returns all of the enum instances and `getWidth` returns the width of the hardware type.

## Workarounds

As of 2/26/2021, the width of the values is always inferred. To work around this, you can add an extra `Value` that forces the width that is desired. This is shown in the example below, where we add a field `ukn` to force the width to be 3 bits wide: 

```scala mdoc
object StoreFunct3 extends ChiselEnum {
    val sb, sh, sw = Value
    val ukn = Value(7.U)
    /** How the values will be mapped
        "sb" -> 0.U,
        "sh" -> 1.U,
        "sw" -> 2.U
    */
}
```

Signed values are not supported so if you want the value signed, you must cast the UInt with `.asSInt`.

## Additional Resources

The ChiselEnum type is much more powerful than stated above. It allows for Sequence, Vec, and Bundle assignments, as well as a `.next` operation to allow for stepping through sequential states and an `.isValid` for checking that a hardware value is a valid `Value`. The source code for the ChiselEnum can be found [here](https://github.com/chipsalliance/chisel3/blob/2a96767097264eade18ff26e1d8bce192383a190/core/src/main/scala/chisel3/StrongEnum.scala) in the class `EnumFactory`. Examples of the ChiselEnum operations can be found [here](https://github.com/chipsalliance/chisel3/blob/dd6871b8b3f2619178c2a333d9d6083805d99e16/src/test/scala/chiselTests/StrongEnum.scala).
