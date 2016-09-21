// See LICENSE for license details.

package object chisel3 {    // scalastyle:ignore package.object.name
  import scala.language.experimental.macros

  import internal.firrtl.Width
  import internal.sourceinfo.{SourceInfo, SourceInfoTransform}
  import util.BitPat

  import chisel3.core.{Binding, FlippedBinder}
  import chisel3.util._
  import chisel3.internal.firrtl.Port

  type Direction = chisel3.core.Direction
  val Input   = chisel3.core.Input
  val Output  = chisel3.core.Output
  val Flipped = chisel3.core.Flipped

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

  type Printable = chisel3.core.Printable
  val Printable = chisel3.core.Printable
  type Printables = chisel3.core.Printables
  val Printables = chisel3.core.Printables
  type PString = chisel3.core.PString
  val PString = chisel3.core.PString
  type FirrtlFormat = chisel3.core.FirrtlFormat
  val FirrtlFormat = chisel3.core.FirrtlFormat
  type Decimal = chisel3.core.Decimal
  val Decimal = chisel3.core.Decimal
  type Hexadecimal = chisel3.core.Hexadecimal
  val Hexadecimal = chisel3.core.Hexadecimal
  type Binary = chisel3.core.Binary
  val Binary = chisel3.core.Binary
  type Character = chisel3.core.Character
  val Character = chisel3.core.Character
  type Name = chisel3.core.Name
  val Name = chisel3.core.Name
  type FullName = chisel3.core.FullName
  val FullName = chisel3.core.FullName
  val Percent = chisel3.core.Percent

  /** Implicit for custom Printable string interpolator */
  implicit class PrintableHelper(val sc: StringContext) extends AnyVal {
    /** Custom string interpolator for generating Printables: p"..."
      * Will call .toString on any non-Printable arguments (mimicking s"...")
      */
    def p(args: Any*): Printable = {
      sc.checkLengths(args) // Enforce sc.parts.size == pargs.size + 1
      val pargs: Seq[Option[Printable]] = args map {
        case p: Printable => Some(p)
        case d: Data => Some(d.toPrintable)
        case any => for {
          v <- Option(any) // Handle null inputs
          str = v.toString
          if !str.isEmpty // Handle empty Strings
        } yield PString(str)
      }
      val parts = sc.parts map StringContext.treatEscapes
      // Zip sc.parts and pargs together ito flat Seq
      // eg. Seq(sc.parts(0), pargs(0), sc.parts(1), pargs(1), ...)
      val seq = for { // append None because sc.parts.size == pargs.size + 1
        (literal, arg) <- parts zip (pargs :+ None)
        optPable <- Seq(Some(PString(literal)), arg)
        pable <- optPable // Remove Option[_]
      } yield pable
      Printables(seq)
    }
  }

  implicit def string2Printable(str: String): Printable = PString(str)

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
    def U: UInt = UInt(BigInt(x), Width())    // scalastyle:ignore method.name
    def S: SInt = SInt(BigInt(x), Width())    // scalastyle:ignore method.name

    def asUInt(): UInt = UInt(x, Width())
    def asSInt(): SInt = SInt(x, Width())
    def asUInt(width: Int): UInt = UInt(x, width)
    def asSInt(width: Int): SInt = SInt(x, width)
  }

  implicit class fromBigIntToLiteral(val x: BigInt) extends AnyVal {
    def U: UInt = UInt(x, Width())    // scalastyle:ignore method.name
    def S: SInt = SInt(x, Width())    // scalastyle:ignore method.name

    def asUInt(): UInt = UInt(x, Width())
    def asSInt(): SInt = SInt(x, Width())
    def asUInt(width: Int): UInt = UInt(x, width)
    def asSInt(width: Int): SInt = SInt(x, width)
  }
  implicit class fromStringToLiteral(val x: String) extends AnyVal {
    def U: UInt = UInt(x)    // scalastyle:ignore method.name
  }
  implicit class fromBooleanToLiteral(val x: Boolean) extends AnyVal {
    def B: Bool = Bool(x)    // scalastyle:ignore method.name
  }

  implicit class fromUIntToBitPatComparable(val x: UInt) extends AnyVal {
    final def === (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def != (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: BitPat): Bool = macro SourceInfoTransform.thatArg

    def do_=== (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x    // scalastyle:ignore method.name
    def do_!= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that != x      // scalastyle:ignore method.name
    def do_=/= (that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x    // scalastyle:ignore method.name
  }

  // Compatibility with existing code.
  val INPUT = chisel3.core.Direction.Input
  val OUTPUT = chisel3.core.Direction.Output
  val NODIR = chisel3.core.Direction.Unspecified
  type ChiselException = chisel3.internal.ChiselException

  class EnqIO[+T <: Data](gen: T) extends DecoupledIO(gen) {
    def init(): Unit = {
      this.noenq()
    }
    override def cloneType: this.type = EnqIO(gen).asInstanceOf[this.type]
  }
  class DeqIO[+T <: Data](gen: T) extends DecoupledIO(gen) {
    val Data = chisel3.core.Data
    Data.setFirrtlDirection(this, Data.getFirrtlDirection(this).flip)
    Binding.bind(this, FlippedBinder, "Error: Cannot flip ")
    def init(): Unit = {
      this.nodeq()
    }
    override def cloneType: this.type = DeqIO(gen).asInstanceOf[this.type]
  }
  object EnqIO {
    def apply[T<:Data](gen: T): EnqIO[T] = new EnqIO(gen)
  }
  object DeqIO {
    def apply[T<:Data](gen: T): DeqIO[T] = new DeqIO(gen)
  }

  // Debugger/Tester access to internal Chisel data structures and methods.
  def getDataElements(a: Aggregate): Seq[Element] = {
    a.allElements
  }
  def getModulePorts(m: Module): Seq[Port] = m.getPorts
  def getFirrtlDirection(d: Data): Direction = chisel3.core.Data.getFirrtlDirection(d)
}
