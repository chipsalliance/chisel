// See LICENSE for license details.

// Allows legacy users to continue using Chisel (capital C) package name while
// moving to the more standard package naming convention chisel3 (lowercase c).

package object Chisel {     // scalastyle:ignore package.object.name
  implicit val defaultCompileOptions = chisel3.core.ExplicitCompileOptions.NotStrict
  type Direction = chisel3.core.Direction
  val INPUT = chisel3.core.Direction.Input
  val OUTPUT = chisel3.core.Direction.Output
  val NODIR = chisel3.core.Direction.Unspecified
  object Flipped {
    def apply[T<:Data](target: T): T = chisel3.core.Flipped[T](target)
  }
  // TODO: Possibly move the AddDirectionToData class here?
  implicit class AddDirMethodToData[T<:Data](val target: T) extends AnyVal {
    def dir: Direction = {
      target match {
        case e: Element => e.dir
        case _ => chisel3.core.Direction.Unspecified
      }
    }
  }

  type ChiselException = chisel3.internal.ChiselException

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
  val stop = chisel3.core.stop

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

  import chisel3.internal.firrtl.Width
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
    def U: UInt = UInt(x, Width())       // scalastyle:ignore method.name
    def S: SInt = SInt(x, Width())       // scalastyle:ignore method.name

    def asUInt(): UInt = UInt(x, Width())
    def asSInt(): SInt = SInt(x, Width())
    def asUInt(width: Int): UInt = UInt(x, width)
    def asSInt(width: Int): SInt = SInt(x, width)
  }
  implicit class fromStringToLiteral(val x: String) extends AnyVal {
    def U: UInt = UInt(x)       // scalastyle:ignore method.name
  }
  implicit class fromBooleanToLiteral(val x: Boolean) extends AnyVal {
    def B: Bool = Bool(x)       // scalastyle:ignore method.name
  }


  type BackendCompilationUtilities = chisel3.BackendCompilationUtilities
  val Driver = chisel3.Driver
  val ImplicitConversions = chisel3.util.ImplicitConversions
  val chiselMain = chisel3.compatibility.chiselMain
  val throwException = chisel3.compatibility.throwException
  val debug = chisel3.compatibility.debug

  object testers {    // scalastyle:ignore object.name
    type BasicTester = chisel3.testers.BasicTester
    val TesterDriver = chisel3.testers.TesterDriver
  }


  val log2Up = chisel3.util.log2Up
  val log2Ceil = chisel3.util.log2Ceil
  val log2Down = chisel3.util.log2Down
  val log2Floor = chisel3.util.log2Floor
  val isPow2 = chisel3.util.isPow2

  val BitPat = chisel3.util.BitPat
  type BitPat = chisel3.util.BitPat

  type ArbiterIO[T <: Data] = chisel3.util.ArbiterIO[T]
  type LockingArbiterLike[T <: Data] = chisel3.util.LockingArbiterLike[T]
  type LockingRRArbiter[T <: Data] = chisel3.util.LockingRRArbiter[T]
  type LockingArbiter[T <: Data] = chisel3.util.LockingArbiter[T]
  type RRArbiter[T <: Data] = chisel3.util.RRArbiter[T]
  type Arbiter[T <: Data] = chisel3.util.Arbiter[T]

  val FillInterleaved = chisel3.util.FillInterleaved
  val PopCount = chisel3.util.PopCount
  val Fill = chisel3.util.Fill
  val Reverse = chisel3.util.Reverse

  val Cat = chisel3.util.Cat

  val Log2 = chisel3.util.Log2

  val unless = chisel3.util.unless
  type SwitchContext[T <: Bits] = chisel3.util.SwitchContext[T]
  val is = chisel3.util.is
  val switch = chisel3.util.switch

  type Counter = chisel3.util.Counter
  val Counter = chisel3.util.Counter

  type DecoupledIO[+T <: Data] = chisel3.util.DecoupledIO[T]
  val DecoupledIO = chisel3.util.Decoupled
  val Decoupled = chisel3.util.Decoupled
  type QueueIO[T <: Data] = chisel3.util.QueueIO[T]
  type Queue[T <: Data] = chisel3.util.Queue[T]
  val Queue = chisel3.util.Queue

  val Enum = chisel3.util.Enum

  val LFSR16 = chisel3.util.LFSR16

  val ListLookup = chisel3.util.ListLookup
  val Lookup = chisel3.util.Lookup

  val Mux1H = chisel3.util.Mux1H
  val PriorityMux = chisel3.util.PriorityMux
  val MuxLookup = chisel3.util.MuxLookup
  val MuxCase = chisel3.util.MuxCase

  val OHToUInt = chisel3.util.OHToUInt
  val PriorityEncoder = chisel3.util.PriorityEncoder
  val UIntToOH = chisel3.util.UIntToOH
  val PriorityEncoderOH = chisel3.util.PriorityEncoderOH

  val RegNext = chisel3.util.RegNext
  val RegInit = chisel3.util.RegInit
  val RegEnable = chisel3.util.RegEnable
  val ShiftRegister = chisel3.util.ShiftRegister

  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  val Valid = chisel3.util.Valid
  val Pipe = chisel3.util.Pipe
  type Pipe[T <: Data] = chisel3.util.Pipe[T]

}
