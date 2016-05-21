// See LICENSE for license details.

// Allows legacy users to continue using Chisel (capital C) package name while
// moving to the more standard package naming convention chisel (lowercase c).

package object Chisel {
  type Direction = chisel.Direction
  val INPUT = chisel.INPUT
  val OUTPUT = chisel.OUTPUT
  val NO_DIR = chisel.NO_DIR
  val debug = chisel.debug
  type Flipped = chisel.Flipped
  type Data = chisel.Data
  val Wire = chisel.Wire
  val Clock = chisel.Clock
  type Clock = chisel.Clock

  type Aggregate = chisel.Aggregate
  val Vec = chisel.Vec
  type Vec[T <: Data] = chisel.Vec[T]
  type VecLike[T <: Data] = chisel.VecLike[T]
  type Bundle = chisel.Bundle

  val assert = chisel.assert

  val BitPat = chisel.BitPat
  type BitPat = chisel.BitPat

  type Bits = chisel.Bits
  val Bits = chisel.Bits
  type Num[T <: Data] = chisel.Num[T]
  type UInt = chisel.UInt
  val UInt = chisel.UInt
  type SInt = chisel.SInt
  val SInt = chisel.SInt
  type Bool = chisel.Bool
  val Bool = chisel.Bool
  val Mux = chisel.Mux

  type BlackBox = chisel.BlackBox

  val Mem = chisel.Mem
  type MemBase[T <: Data] = chisel.MemBase[T]
  type Mem[T <: Data] = chisel.Mem[T]
  val SeqMem = chisel.SeqMem
  type SeqMem[T <: Data] = chisel.SeqMem[T]

  val Module = chisel.Module
  type Module = chisel.Module

  val printf = chisel.printf

  val Reg = chisel.Reg

  val when = chisel.when
  type WhenContext = chisel.WhenContext


  type BackendCompilationUtilities = chisel.BackendCompilationUtilities
  val Driver = chisel.Driver
  type FileSystemUtilities = chisel.FileSystemUtilities
  val ImplicitConversions = chisel.ImplicitConversions
  val chiselMain = chisel.chiselMain
  val throwException = chisel.throwException


  val log2Up = chisel.log2Up
  val log2Ceil = chisel.log2Ceil
  val log2Down = chisel.log2Down
  val log2Floor = chisel.log2Floor
  val isPow2 = chisel.isPow2

  type ArbiterIO[T <: Data] = chisel.ArbiterIO[T]
  type LockingArbiterLike[T <: Data] = chisel.LockingArbiterLike[T]
  type LockingRRArbiter[T <: Data] = chisel.LockingRRArbiter[T]
  type LockingArbiter[T <: Data] = chisel.LockingArbiter[T]
  type RRArbiter[T <: Data] = chisel.RRArbiter[T]
  type Arbiter[T <: Data] = chisel.Arbiter[T]

  val FillInterleaved = chisel.FillInterleaved
  val PopCount = chisel.PopCount
  val Fill = chisel.Fill
  val Reverse = chisel.Reverse

  val Cat = chisel.Cat

  val Log2 = chisel.Log2

  val unless = chisel.unless
  type SwitchContext[T <: Bits] = chisel.SwitchContext[T]
  val is = chisel.is
  val switch = chisel.switch

  type Counter = chisel.Counter
  val Counter = chisel.Counter

  type DecoupledIO[+T <: Data] = chisel.DecoupledIO[T]
  val Decoupled = chisel.Decoupled
  type EnqIO[T <: Data] = chisel.EnqIO[T]
  type DeqIO[T <: Data] = chisel.DeqIO[T]
  type DecoupledIOC[+T <: Data] = chisel.DecoupledIOC[T]
  type QueueIO[T <: Data] = chisel.QueueIO[T]
  type Queue[T <: Data] = chisel.Queue[T]
  val Queue = chisel.Queue

  val Enum = chisel.Enum

  val LFSR16 = chisel.LFSR16

  val ListLookup = chisel.ListLookup
  val Lookup = chisel.Lookup

  val Mux1H = chisel.Mux1H
  val PriorityMux = chisel.PriorityMux
  val MuxLookup = chisel.MuxLookup
  val MuxCase = chisel.MuxCase

  val OHToUInt = chisel.OHToUInt
  val PriorityEncoder = chisel.PriorityEncoder
  val UIntToOH = chisel.UIntToOH
  val PriorityEncoderOH = chisel.PriorityEncoderOH

  val RegNext = chisel.RegNext
  val RegInit = chisel.RegInit
  val RegEnable = chisel.RegEnable
  val ShiftRegister = chisel.ShiftRegister

  type ValidIO[+T <: Data] = chisel.ValidIO[T]
  val Valid = chisel.Valid
  val Pipe = chisel.Pipe
  type Pipe[T <: Data] = chisel.Pipe[T]
}

package Chisel {
  package object testers {
    type BasicTester = chisel.testers.BasicTester
    val TesterDriver = chisel.testers.TesterDriver
  }
}
