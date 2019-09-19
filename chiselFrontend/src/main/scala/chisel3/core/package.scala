// See LICENSE for license details.

package chisel3

/**
 * These definitions exist to deal with those clients that relied on chisel3.core.
 * They are deprecated and will be removed in the future.
 */
package object core {

  @deprecated("Use the version in chisel3._", "3.2")
  val CompileOptions = chisel3.CompileOptions

  @deprecated("Use the version in chisel3._", "3.2")
  val Input   = chisel3.Input
  @deprecated("Use the version in chisel3._", "3.2")
  val Output  = chisel3.Output
  @deprecated("Use the version in chisel3._", "3.2")
  val Flipped = chisel3.Flipped
  @deprecated("Use the version in chisel3._", "3.2")
  val chiselTypeOf = chisel3.chiselTypeOf

  @deprecated("Use the version in chisel3._", "3.2")
  type Data = chisel3.Data

  @deprecated("Use the version in chisel3._", "3.2")
  val WireDefault = chisel3.WireDefault

  @deprecated("Use the version in chisel3._", "3.2")
  val Clock = chisel3.Clock
  @deprecated("Use the version in chisel3._", "3.2")
  type Clock = chisel3.Clock

  @deprecated("Use the version in chisel3._", "3.2")
  type Reset = chisel3.Reset

  @deprecated("Use the version in chisel3._", "3.2")
  type Aggregate = chisel3.Aggregate

  @deprecated("Use the version in chisel3._", "3.2")
  val Vec = chisel3.Vec
  @deprecated("Use the version in chisel3._", "3.2")
  val VecInit = chisel3.VecInit
  @deprecated("Use the version in chisel3._", "3.2")
  type Vec[T <: Data] = chisel3.Vec[T]
  @deprecated("Use the version in chisel3._", "3.2")
  type VecLike[T <: Data] = chisel3.VecLike[T]
  @deprecated("Use the version in chisel3._", "3.2")
  type Bundle = chisel3.Bundle
  @deprecated("Use the version in chisel3._", "3.2")
  type IgnoreSeqInBundle = chisel3.IgnoreSeqInBundle
  @deprecated("Use the version in chisel3._", "3.2")
  type Record = chisel3.Record

  @deprecated("Use the version in chisel3._", "3.2")
  val assert = chisel3.assert

  @deprecated("Use the version in chisel3._", "3.2")
  type Element = chisel3.Element
  @deprecated("Use the version in chisel3._", "3.2")
  type Bits = chisel3.Bits

  // These provide temporary compatibility for those who foolishly imported from chisel3.core
  @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
    " Use chisel3.RawModule instead. This alias will be removed in 3.3.", "since the beginning of time")
  type RawModule = chisel3.RawModule
  @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
    "Use chisel3.MultiIOModule instead. This alias will be removed in 3.3.", "since the beginning of time")
  type MultiIOModule = chisel3.MultiIOModule
  @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
    " Use chisel3.RawModule instead. This alias will be removed in 3.3.", "since the beginning of time")
  type UserModule = chisel3.RawModule
  @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
    "Use chisel3.MultiIOModule instead. This alias will be removed in 3.3.", "since the beginning of time")
  type ImplicitModule = chisel3.MultiIOModule

  @deprecated("Use the version in chisel3._", "3.2")
  val Bits = chisel3.Bits
  @deprecated("Use the version in chisel3._", "3.2")
  type Num[T <: chisel3.Data] = chisel3.Num[T]
  @deprecated("Use the version in chisel3._", "3.2")
  type UInt = chisel3.UInt
  @deprecated("Use the version in chisel3._", "3.2")
  val UInt = chisel3.UInt
  @deprecated("Use the version in chisel3._", "3.2")
  type SInt = chisel3.SInt
  @deprecated("Use the version in chisel3._", "3.2")
  val SInt = chisel3.SInt
  @deprecated("Use the version in chisel3._", "3.2")
  type Bool = chisel3.Bool
  @deprecated("Use the version in chisel3._", "3.2")
  val Bool = chisel3.Bool
  @deprecated("Use the version in chisel3._", "3.2")
  val Mux = chisel3.Mux

  @deprecated("Use the version in chisel3._", "3.2")
  type BlackBox = chisel3.BlackBox

  @deprecated("Use the version in chisel3._", "3.2")
  val Mem = chisel3.Mem
  @deprecated("Use the version in chisel3._", "3.2")
  type MemBase[T <: chisel3.Data] = chisel3.MemBase[T]
  @deprecated("Use the version in chisel3._", "3.2")
  type Mem[T <: chisel3.Data] = chisel3.Mem[T]
  @deprecated("Use the version in chisel3._", "3.2")
  val SyncReadMem = chisel3.SyncReadMem
  @deprecated("Use the version in chisel3._", "3.2")
  type SyncReadMem[T <: chisel3.Data] = chisel3.SyncReadMem[T]

  @deprecated("Use the version in chisel3._", "3.2")
  val Module = chisel3.Module
  @deprecated("Use the version in chisel3._", "3.2")
  type Module = chisel3.Module

  @deprecated("Use the version in chisel3._", "3.2")
  val printf = chisel3.printf

  @deprecated("Use the version in chisel3._", "3.2")
  val RegNext = chisel3.RegNext
  @deprecated("Use the version in chisel3._", "3.2")
  val RegInit = chisel3.RegInit
  @deprecated("Use the version in chisel3._", "3.2")
  val Reg = chisel3.Reg

  @deprecated("Use the version in chisel3._", "3.2")
  val when = chisel3.when
  @deprecated("Use the version in chisel3._", "3.2")
  type WhenContext = chisel3.WhenContext

  @deprecated("Use the version in chisel3._", "3.2")
  type Printable = chisel3.Printable
  @deprecated("Use the version in chisel3._", "3.2")
  val Printable = chisel3.Printable
  @deprecated("Use the version in chisel3._", "3.2")
  type Printables = chisel3.Printables
  @deprecated("Use the version in chisel3._", "3.2")
  val Printables = chisel3.Printables
  @deprecated("Use the version in chisel3._", "3.2")
  type PString = chisel3.PString
  @deprecated("Use the version in chisel3._", "3.2")
  val PString = chisel3.PString
  @deprecated("Use the version in chisel3._", "3.2")
  type FirrtlFormat = chisel3.FirrtlFormat
  @deprecated("Use the version in chisel3._", "3.2")
  val FirrtlFormat = chisel3.FirrtlFormat
  @deprecated("Use the version in chisel3._", "3.2")
  type Decimal = chisel3.Decimal
  @deprecated("Use the version in chisel3._", "3.2")
  val Decimal = chisel3.Decimal
  @deprecated("Use the version in chisel3._", "3.2")
  type Hexadecimal = chisel3.Hexadecimal
  val Hexadecimal = chisel3.Hexadecimal
  @deprecated("Use the version in chisel3._", "3.2")
  type Binary = chisel3.Binary
  @deprecated("Use the version in chisel3._", "3.2")
  val Binary = chisel3.Binary
  @deprecated("Use the version in chisel3._", "3.2")
  type Character = chisel3.Character
  @deprecated("Use the version in chisel3._", "3.2")
  val Character = chisel3.Character
  @deprecated("Use the version in chisel3._", "3.2")
  type Name = chisel3.Name
  @deprecated("Use the version in chisel3._", "3.2")
  val Name = chisel3.Name
  @deprecated("Use the version in chisel3._", "3.2")
  type FullName = chisel3.FullName
  @deprecated("Use the version in chisel3._", "3.2")
  val FullName = chisel3.FullName
  @deprecated("Use the version in chisel3._", "3.2")
  val Percent = chisel3.Percent

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type Param = chisel3.experimental.Param
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type IntParam = chisel3.experimental.IntParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val IntParam = chisel3.experimental.IntParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type DoubleParam = chisel3.experimental.DoubleParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val DoubleParam = chisel3.experimental.DoubleParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type StringParam = chisel3.experimental.StringParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val StringParam = chisel3.experimental.StringParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type RawParam = chisel3.experimental.RawParam
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val RawParam = chisel3.experimental.RawParam

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type Analog = chisel3.experimental.Analog
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val Analog = chisel3.experimental.Analog

  @deprecated("Use the version in chisel3._", "3.2")
  implicit class fromIntToWidth(int: Int) extends chisel3.fromIntToWidth(int)

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val attach = chisel3.experimental.attach

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type EnumType = chisel3.experimental.EnumType
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type EnumFactory = chisel3.experimental.EnumFactory
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val EnumAnnotations = chisel3.experimental.EnumAnnotations

  @deprecated("Use the version in chisel3._", "3.2")
  val withClockAndReset = chisel3.withClockAndReset
  @deprecated("Use the version in chisel3._", "3.2")
  val withClock = chisel3.withClock
  @deprecated("Use the version in chisel3._", "3.2")
  val withReset = chisel3.withReset

  @deprecated("Use the version in chisel3._", "3.2")
  val dontTouch = chisel3.dontTouch

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type BaseModule = chisel3.experimental.BaseModule
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type ExtModule = chisel3.experimental.ExtModule

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val IO = chisel3.experimental.IO

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type FixedPoint = chisel3.experimental.FixedPoint
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val FixedPoint = chisel3.experimental.FixedPoint
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  implicit class fromDoubleToLiteral(double: Double) extends experimental.FixedPoint.Implicits.fromDoubleToLiteral(double)
  @deprecated("Use the version in chisel3.experimental._", "3.2")
  implicit class fromIntToBinaryPoint(int: Int) extends experimental.FixedPoint.Implicits.fromIntToBinaryPoint(int)

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  type RunFirrtlTransform = chisel3.experimental.RunFirrtlTransform

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val annotate = chisel3.experimental.annotate

  @deprecated("Use the version in chisel3.experimental._", "3.2")
  val DataMirror = chisel3.experimental.DataMirror
  @deprecated("Use the version in chisel3._", "3.2")
  type ActualDirection = chisel3.ActualDirection
  @deprecated("Use the version in chisel3._", "3.2")
  val ActualDirection = chisel3.ActualDirection

  @deprecated("Use the version in chisel3.internal._", "3.2")
  val requireIsHardware = chisel3.internal.requireIsHardware
  @deprecated("Use the version in chisel3.internal._", "3.2")
  val requireIsChiselType = chisel3.internal.requireIsChiselType
  @deprecated("Use the version in chisel3.internal._", "3.2")
  val BiConnect = chisel3.internal.BiConnect
  @deprecated("Use the version in chisel3.internal._", "3.2")
  val MonoConnect = chisel3.internal.MonoConnect
  @deprecated("Use the version in chisel3.internal._", "3.2")
  val BindingDirection = chisel3.internal.BindingDirection
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type Binding = chisel3.internal.Binding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type TopBinding = chisel3.internal.TopBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type UnconstrainedBinding = chisel3.internal.UnconstrainedBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type ConstrainedBinding = chisel3.internal.ConstrainedBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type ReadOnlyBinding = chisel3.internal.ReadOnlyBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type OpBinding = chisel3.internal.OpBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type MemoryPortBinding = chisel3.internal.MemoryPortBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type PortBinding = chisel3.internal.PortBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type RegBinding = chisel3.internal.RegBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type WireBinding = chisel3.internal.WireBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type ChildBinding = chisel3.internal.ChildBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type DontCareBinding = chisel3.internal.DontCareBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type LitBinding = chisel3.internal.LitBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type ElementLitBinding = chisel3.internal.ElementLitBinding
  @deprecated("Use the version in chisel3.internal._", "3.2")
  type BundleLitBinding = chisel3.internal.BundleLitBinding
}
