package object chisel3 {
  import scala.language.experimental.macros

  import internal.firrtl.Width
  import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

  import util.BitPat


  type Direction = chisel3.core.Direction
  object Input {
    def apply[T<:Data](target: T): T = chisel3.core.Input(target)
  }
  object Output {
    def apply[T<:Data](target: T): T = chisel3.core.Output(target)
  }
  object Flipped {
    def apply[T<:Data](target: T): T = chisel3.core.Flipped(target)
  }

  type Data = chisel3.core.Data
  val Wire = chisel3.core.Wire
  val Clock = chisel3.core.Clock
  type Clock = chisel3.core.Clock

  type Aggregate = chisel3.core.Aggregate
  val Vec = chisel3.core.Vec
  type Vec[T <: Data] = chisel3.core.Vec[T]
  type VecLike[T <: Data] = chisel3.core.VecLike[T]
  type Bundle = chisel3.core.Bundle

  val assert = chisel3.core.assert

  type Element = chisel3.core.Element
  type Bits = chisel3.core.Bits
  val Bits = chisel3.core.Bits
  type Num[T <: Data] = chisel3.core.Num[T]
  type UInt = chisel3.core.UInt
  val UInt = chisel3.core.UInt
  type SInt = chisel3.core.SInt
  val SInt = chisel3.core.SInt
  type Bool = chisel3.core.Bool
  val Bool = chisel3.core.Bool
  val Mux = chisel3.core.Mux

  type BlackBox = chisel3.core.BlackBox

  val Mem = chisel3.core.Mem
  type MemBase[T <: Data] = chisel3.core.MemBase[T]
  type Mem[T <: Data] = chisel3.core.Mem[T]
  val SeqMem = chisel3.core.SeqMem
  type SeqMem[T <: Data] = chisel3.core.SeqMem[T]

  val Module = chisel3.core.Module
  type Module = chisel3.core.Module

  val printf = chisel3.core.printf

  val Reg = chisel3.core.Reg

  val when = chisel3.core.when
  type WhenContext = chisel3.core.WhenContext

  /**
  * These implicit classes allow one to convert scala.Int|scala.BigInt to
  * Chisel.UInt|Chisel.SInt by calling .asUInt|.asSInt on them, respectively.
  * The versions .asUInt(width)|.asSInt(width) are also available to explicitly
  * mark a width for the new literal.
  *
  * Also provides .asBool to scala.Boolean and .asUInt to String
  *
  * Note that, for stylistic reasons, one should avoid extracting immediately
  * after this call using apply, ie. 0.asUInt(1)(0) due to potential for
  * confusion (the 1 is a bit length and the 0 is a bit extraction position).
  * Prefer storing the result and then extracting from it.
  */
  implicit class fromIntToLiteral(val x: Int) extends AnyVal {
    def U: UInt = UInt(BigInt(x), Width())
    def S: SInt = SInt(BigInt(x), Width())

    def asUInt() = UInt(x, Width())
    def asSInt() = SInt(x, Width())
    def asUInt(width: Int) = UInt(x, width)
    def asSInt(width: Int) = SInt(x, width)
  }
  
  implicit class fromBigIntToLiteral(val x: BigInt) extends AnyVal {
    def U: UInt = UInt(x, Width())
    def S: SInt = SInt(x, Width())
  }
  implicit class fromStringToLiteral(val x: String) extends AnyVal {
    def U: UInt = UInt(x)
  }
  implicit class fromBooleanToLiteral(val x: Boolean) extends AnyVal {
    def B: Bool = Bool(x)
  }
  
  implicit class fromUIntToBitPatComparable(val x: UInt) extends AnyVal {
    final def === (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def != (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: BitPat): Bool = macro SourceInfoTransform.thatArg

    def do_=== (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x
    def do_!= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that != x
    def do_=/= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
  }

  val INPUT = chisel3.core.Direction.Input
  val OUTPUT = chisel3.core.Direction.Output
}
