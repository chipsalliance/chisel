package object chisel3 {
  import scala.language.experimental.macros

  import internal.firrtl.Width
  import internal.sourceinfo.{SourceInfo, SourceInfoTransform}
  import util.BitPat


  type Direction = chisel3.core.Direction
  val INPUT = chisel3.core.INPUT
  val OUTPUT = chisel3.core.OUTPUT
  val NO_DIR = chisel3.core.NO_DIR
  type Flipped = chisel3.core.Flipped
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
