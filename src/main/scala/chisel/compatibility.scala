// See LICENSE for license details.

// Allows legacy users to continue using Chisel (capital C) package name while
// moving to the more standard package naming convention chisel (lowercase c).

package object Chisel {
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
  val stop = chisel.core.stop

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


  type BackendCompilationUtilities = chisel.BackendCompilationUtilities
  val Driver = chisel.Driver
  type FileSystemUtilities = chisel.compatibility.FileSystemUtilities
  val ImplicitConversions = chisel.util.ImplicitConversions
  val chiselMain = chisel.compatibility.chiselMain
  val throwException = chisel.compatibility.throwException
  val debug = chisel.compatibility.debug

  object testers {
    type BasicTester = chisel.testers.BasicTester
    val TesterDriver = chisel.testers.TesterDriver
  }


  val log2Up = chisel.util.log2Up
  val log2Ceil = chisel.util.log2Ceil
  val log2Down = chisel.util.log2Down
  val log2Floor = chisel.util.log2Floor
  val isPow2 = chisel.util.isPow2

  val BitPat = chisel.util.BitPat
  type BitPat = chisel.util.BitPat

  type ArbiterIO[T <: Data] = chisel.util.ArbiterIO[T]
  type LockingArbiterLike[T <: Data] = chisel.util.LockingArbiterLike[T]
  type LockingRRArbiter[T <: Data] = chisel.util.LockingRRArbiter[T]
  type LockingArbiter[T <: Data] = chisel.util.LockingArbiter[T]
  type RRArbiter[T <: Data] = chisel.util.RRArbiter[T]
  type Arbiter[T <: Data] = chisel.util.Arbiter[T]

  val FillInterleaved = chisel.util.FillInterleaved
  val PopCount = chisel.util.PopCount
  val Fill = chisel.util.Fill
  val Reverse = chisel.util.Reverse

  val Cat = chisel.util.Cat

  val Log2 = chisel.util.Log2

  val unless = chisel.util.unless
  type SwitchContext[T <: Bits] = chisel.util.SwitchContext[T]
  val is = chisel.util.is
  val switch = chisel.util.switch

  type Counter = chisel.util.Counter
  val Counter = chisel.util.Counter

  type DecoupledIO[+T <: Data] = chisel.util.DecoupledIO[T]
  val Decoupled = chisel.util.Decoupled
  type EnqIO[T <: Data] = chisel.util.EnqIO[T]
  type DeqIO[T <: Data] = chisel.util.DeqIO[T]
  type DecoupledIOC[+T <: Data] = chisel.util.DecoupledIOC[T]
  type QueueIO[T <: Data] = chisel.util.QueueIO[T]
  type Queue[T <: Data] = chisel.util.Queue[T]
  val Queue = chisel.util.Queue

  val Enum = chisel.util.Enum

  val LFSR16 = chisel.util.LFSR16

  val ListLookup = chisel.util.ListLookup
  val Lookup = chisel.util.Lookup

  val Mux1H = chisel.util.Mux1H
  val PriorityMux = chisel.util.PriorityMux
  val MuxLookup = chisel.util.MuxLookup
  val MuxCase = chisel.util.MuxCase

  val OHToUInt = chisel.util.OHToUInt
  val PriorityEncoder = chisel.util.PriorityEncoder
  val UIntToOH = chisel.util.UIntToOH
  val PriorityEncoderOH = chisel.util.PriorityEncoderOH

  val RegNext = chisel.util.RegNext
  val RegInit = chisel.util.RegInit
  val RegEnable = chisel.util.RegEnable
  val ShiftRegister = chisel.util.ShiftRegister

  type ValidIO[+T <: Data] = chisel.util.ValidIO[T]
  val Valid = chisel.util.Valid
  val Pipe = chisel.util.Pipe
  type Pipe[T <: Data] = chisel.util.Pipe[T]


  import chisel.internal.firrtl.Width
  implicit def fromBigIntToLiteral(x: BigInt): chisel.fromBigIntToLiteral =
    new chisel.fromBigIntToLiteral(x)
  implicit def fromIntToLiteral(x: Int): chisel.fromIntToLiteral=
    new chisel.fromIntToLiteral(x)
  implicit def fromStringToLiteral(x: String): chisel.fromStringToLiteral=
    new chisel.fromStringToLiteral(x)
  implicit def fromBooleanToLiteral(x: Boolean): chisel.fromBooleanToLiteral=
    new chisel.fromBooleanToLiteral(x)
}
