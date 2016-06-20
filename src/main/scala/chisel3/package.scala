package object chisel {
  import scala.language.experimental.macros

  import internal.firrtl.Width
  import internal.sourceinfo.{SourceInfo, SourceInfoTransform}
  import util.BitPat


  type Direction = chisel.core.Direction
  val INPUT = chisel.core.INPUT
  val OUTPUT = chisel.core.OUTPUT
  val NO_DIR = chisel.core.NO_DIR
  type Flipped = chisel.core.Flipped
  type Data = chisel.core.Data
  val Wire = chisel.core.Wire
  val Clock = chisel.core.Clock
  type Clock = chisel.core.Clock

  type Aggregate = chisel.core.Aggregate
  val Vec = chisel.core.Vec
  type Vec[T <: Data] = chisel.core.Vec[T]
  type VecLike[T <: Data] = chisel.core.VecLike[T]
  type Bundle = chisel.core.Bundle

  val assert = chisel.core.assert

  type Element = chisel.core.Element
  type Bits = chisel.core.Bits
  val Bits = chisel.core.Bits
  type Num[T <: Data] = chisel.core.Num[T]
  type UInt = chisel.core.UInt
  val UInt = chisel.core.UInt
  type SInt = chisel.core.SInt
  val SInt = chisel.core.SInt
  type Bool = chisel.core.Bool
  val Bool = chisel.core.Bool
  val Mux = chisel.core.Mux

  type BlackBox = chisel.core.BlackBox

  val Mem = chisel.core.Mem
  type MemBase[T <: Data] = chisel.core.MemBase[T]
  type Mem[T <: Data] = chisel.core.Mem[T]
  val SeqMem = chisel.core.SeqMem
  type SeqMem[T <: Data] = chisel.core.SeqMem[T]

  val Module = chisel.core.Module
  type Module = chisel.core.Module

  val printf = chisel.core.printf

  val Reg = chisel.core.Reg

  val when = chisel.core.when
  type WhenContext = chisel.core.WhenContext


  implicit class fromBigIntToLiteral(val x: BigInt) extends AnyVal {
    def U: UInt = UInt(x, Width())
    def S: SInt = SInt(x, Width())
  }
  implicit class fromIntToLiteral(val x: Int) extends AnyVal {
    def U: UInt = UInt(BigInt(x), Width())
    def S: SInt = SInt(BigInt(x), Width())
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
}
