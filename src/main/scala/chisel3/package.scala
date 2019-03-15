// See LICENSE for license details.

import chisel3.core.CompileOptions

/** The chisel3 package contains the chisel3 API.
  * It maps core components into the public chisel3 namespace.
  */
package object chisel3 {    // scalastyle:ignore package.object.name
  import scala.language.implicitConversions

  import internal.firrtl.Width

  import util.BitPat

  import chisel3.util._
  import chisel3.internal.chiselRuntimeDeprecated
  import chisel3.internal.firrtl.Port
  import chisel3.core.CompileOptions

  val Input   = chisel3.core.Input
  val Output  = chisel3.core.Output
  val Flipped = chisel3.core.Flipped
  val chiselTypeOf = chisel3.core.chiselTypeOf

  type Data = chisel3.core.Data
  object Wire extends chisel3.core.WireFactory {
    import chisel3.core.CompileOptions

    @chiselRuntimeDeprecated
    @deprecated("Wire(init=init) is deprecated, use WireDefault(init) instead", "chisel3")
    def apply[T <: Data](dummy: Int = 0, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.core.WireDefault(init)

    @chiselRuntimeDeprecated
    @deprecated("Wire(t, init) is deprecated, use WireDefault(t, init) instead", "chisel3")
    def apply[T <: Data](t: T, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.core.WireDefault(t, init)

    @chiselRuntimeDeprecated
    @deprecated("Wire(t, init) is deprecated, use WireDefault(t, init) instead", "chisel3")
    def apply[T <: Data](t: T, init: DontCare.type)(implicit compileOptions: CompileOptions): T =
      chisel3.core.WireDefault(t, init)
  }
  val WireInit = chisel3.core.WireDefault
  val WireDefault = chisel3.core.WireDefault

  val Clock = chisel3.core.Clock
  type Clock = chisel3.core.Clock

  // Clock and reset scoping functions
  val withClockAndReset = chisel3.core.withClockAndReset
  val withClock = chisel3.core.withClock
  val withReset = chisel3.core.withReset

  implicit class AddDirectionToData[T<:Data](target: T) {
    @chiselRuntimeDeprecated
    @deprecated("Input(Data) should be used over Data.asInput", "chisel3")
    def asInput: T = Input(target)

    @chiselRuntimeDeprecated
    @deprecated("Output(Data) should be used over Data.asOutput", "chisel3")
    def asOutput: T = Output(target)

    @chiselRuntimeDeprecated
    @deprecated("Flipped(Data) should be used over Data.flip", "chisel3")
    def flip(): T = Flipped(target)
  }

  implicit class fromBitsable[T <: Data](data: T) {
    import chisel3.core.CompileOptions
    import chisel3.internal.sourceinfo.SourceInfo

    @chiselRuntimeDeprecated
    @deprecated("fromBits is deprecated, use asTypeOf instead", "chisel3")
    def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      that.asTypeOf(data)
    }
  }

  implicit class cloneTypeable[T <: Data](target: T) {
    @chiselRuntimeDeprecated
    @deprecated("chiselCloneType is deprecated, use chiselTypeOf(...) to get the Chisel Type of a hardware object", "chisel3")
    def chiselCloneType: T = {
      target.cloneTypeFull.asInstanceOf[T]
    }
  }

  type Aggregate = chisel3.core.Aggregate
  object Vec extends chisel3.core.VecFactory {
    import scala.language.experimental.macros
    import chisel3.core.CompileOptions
    import chisel3.internal.sourceinfo._

    @chiselRuntimeDeprecated
    @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
    def apply[T <: Data](gen: T, n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      apply(n, gen)

    @chiselRuntimeDeprecated
    @deprecated("Vec.fill(n)(gen) is deprecated, use VecInit(Seq.fill(n)(gen)) instead", "chisel3")
    def fill[T <: Data](n: Int)(gen: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      apply(Seq.fill(n)(gen))

    def apply[T <: Data](elts: Seq[T]): Vec[T] = macro VecTransform.apply_elts
    @chiselRuntimeDeprecated
    @deprecated("Vec(elts) is deprecated, use VecInit(elts) instead", "chisel3")
    def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.core.VecInit(elts)

    def apply[T <: Data](elt0: T, elts: T*): Vec[T] = macro VecTransform.apply_elt0
    @chiselRuntimeDeprecated
    @deprecated("Vec(elt0, ...) is deprecated, use VecInit(elt0, ...) instead", "chisel3")
    def do_apply[T <: Data](elt0: T, elts: T*)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.core.VecInit(elt0 +: elts.toSeq)

    def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = macro VecTransform.tabulate
    @chiselRuntimeDeprecated
    @deprecated("Vec.tabulate(n)(gen) is deprecated, use VecInit.tabulate(n)(gen) instead", "chisel3")
    def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.core.VecInit.tabulate(n)(gen)
  }
  val VecInit = chisel3.core.VecInit
  type Vec[T <: Data] = chisel3.core.Vec[T]
  type VecLike[T <: Data] = chisel3.core.VecLike[T]
  type Bundle = chisel3.core.Bundle
  type IgnoreSeqInBundle = chisel3.core.IgnoreSeqInBundle
  type Record = chisel3.core.Record

  val assert = chisel3.core.assert

  type Element = chisel3.core.Element
  type Bits = chisel3.core.Bits

  // Some possible regex replacements for the literal specifier deprecation:
  // (note: these are not guaranteed to handle all edge cases! check all replacements!)
  // Bool\((true|false)\)
  //  => $1.B
  // UInt\(width\s*=\s*(\d+|[_a-zA-Z][_0-9a-zA-Z]*)\)
  //  => UInt($1.W)
  // (UInt|SInt|Bits).width\((\d+|[_a-zA-Z][_0-9a-zA-Z]*)\)
  //  => $1($2.W)
  // (U|S)Int\((-?\d+|0[xX][0-9a-fA-F]+)\)
  //  => $2.$1
  // UInt\((\d+|0[xX][0-9a-fA-F]+),\s*(?:width\s*=)?\s*(\d+|[_a-zA-Z][_0-9a-zA-Z]*)\)
  //  => $1.U($2.W)
  // (UInt|SInt|Bool)\(([_a-zA-Z][_0-9a-zA-Z]*)\)
  //  => $2.as$1
  // (UInt|SInt)\(([_a-zA-Z][_0-9a-zA-Z]*),\s*(?:width\s*=)?\s*(\d+|[_a-zA-Z][_0-9a-zA-Z]*)\)
  //  => $2.as$1($3.W)

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    * These will be removed very soon. It's recommended you port your code ASAP.
    */
  trait UIntFactory extends chisel3.core.UIntFactory {
    /** Create a UInt literal with inferred width. */
    @chiselRuntimeDeprecated
    @deprecated("use n.U", "chisel3, will be removed by end of 2017")
    def apply(n: String): UInt = n.asUInt

    /** Create a UInt literal with fixed width. */
    @chiselRuntimeDeprecated
    @deprecated("use n.U(width.W)", "chisel3, will be removed by end of 2017")
    def apply(n: String, width: Int): UInt = n.asUInt(width.W)

    /** Create a UInt literal with specified width. */
    @chiselRuntimeDeprecated
    @deprecated("use value.U(width)", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt, width: Width): UInt = value.asUInt(width)

    /** Create a UInt literal with fixed width. */
    @chiselRuntimeDeprecated
    @deprecated("use value.U(width.W)", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt, width: Int): UInt = value.asUInt(width.W)

    /** Create a UInt literal with inferred width.- compatibility with Chisel2. */
    @chiselRuntimeDeprecated
    @deprecated("use value.U", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt): UInt = value.asUInt

    /** Create a UInt with a specified width */
    @chiselRuntimeDeprecated
    @deprecated("use UInt(width.W)", "chisel3, will be removed by end of 2017")
    def width(width: Int): UInt = apply(width.W)

    /** Create a UInt port with specified width. */
    @chiselRuntimeDeprecated
    @deprecated("use UInt(width)", "chisel3, will be removed by end of 2017")
    def width(width: Width): UInt = apply(width)
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    * These will be removed very soon. It's recommended you move your code soon.
    */
  trait SIntFactory extends chisel3.core.SIntFactory {
    /** Create a SInt type or port with fixed width. */
    @chiselRuntimeDeprecated
    @deprecated("use SInt(width.W)", "chisel3, will be removed by end of 2017")
    def width(width: Int): SInt = apply(width.W)

    /** Create an SInt type with specified width. */
    @chiselRuntimeDeprecated
    @deprecated("use SInt(width)", "chisel3, will be removed by end of 2017")
    def width(width: Width): SInt = apply(width)

    /** Create an SInt literal with inferred width. */
    @chiselRuntimeDeprecated
    @deprecated("use value.S", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt): SInt = value.asSInt

    /** Create an SInt literal with fixed width. */
    @chiselRuntimeDeprecated
    @deprecated("use value.S(width.W)", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt, width: Int): SInt = value.asSInt(width.W)

    /** Create an SInt literal with specified width. */
    @chiselRuntimeDeprecated
    @deprecated("use value.S(width)", "chisel3, will be removed by end of 2017")
    def apply(value: BigInt, width: Width): SInt = value.asSInt(width)

    @chiselRuntimeDeprecated
    @deprecated("use value.S", "chisel3, will be removed by end of 2017")
    def Lit(value: BigInt): SInt = value.asSInt // scalastyle:ignore method.name

    @chiselRuntimeDeprecated
    @deprecated("use value.S(width)", "chisel3, will be removed by end of 2017")
    def Lit(value: BigInt, width: Int): SInt = value.asSInt(width.W) // scalastyle:ignore method.name
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    * These will be removed very soon. It's recommended you move your code soon.
    */
  trait BoolFactory extends chisel3.core.BoolFactory {
    /** Creates Bool literal.
     */
    @chiselRuntimeDeprecated
    @deprecated("use x.B", "chisel3, will be removed by end of 2017")
    def apply(x: Boolean): Bool = x.B
  }

  object Bits extends UIntFactory
  type Num[T <: Data] = chisel3.core.Num[T]
  type UInt = chisel3.core.UInt
  object UInt extends UIntFactory
  type SInt = chisel3.core.SInt
  object SInt extends SIntFactory
  type Bool = chisel3.core.Bool
  object Bool extends BoolFactory
  val Mux = chisel3.core.Mux

  type BlackBox = chisel3.core.BlackBox

  type InstanceId = chisel3.internal.InstanceId

  val Mem = chisel3.core.Mem
  type MemBase[T <: Data] = chisel3.core.MemBase[T]
  type Mem[T <: Data] = chisel3.core.Mem[T]
  val SyncReadMem = chisel3.core.SyncReadMem
  type SyncReadMem[T <: Data] = chisel3.core.SyncReadMem[T]

  @deprecated("Use 'SyncReadMem'", "chisel3")
  val SeqMem = chisel3.core.SyncReadMem
  @deprecated("Use 'SyncReadMem'", "chisel3")
  type SeqMem[T <: Data] = chisel3.core.SyncReadMem[T]

  val Module = chisel3.core.Module
  type Module = chisel3.core.LegacyModule

  val printf = chisel3.core.printf

  val RegNext = chisel3.core.RegNext
  val RegInit = chisel3.core.RegInit
  object Reg {
    import chisel3.core.{Binding, CompileOptions}
    import chisel3.internal.sourceinfo.SourceInfo
    import chisel3.internal.throwException

    // Passthrough for chisel3.core.Reg
    // TODO: make val Reg = chisel3.core.Reg once we eliminate the legacy Reg constructor
    def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      chisel3.core.Reg(t)

    @chiselRuntimeDeprecated
    @deprecated("Use Reg(t), RegNext(next, [init]) or RegInit([t], init) instead", "chisel3")
    def apply[T <: Data](t: T = null, next: T = null, init: T = null)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      if (t ne null) {
        val reg = if (init ne null) {
          RegInit(t, init)
        } else {
          chisel3.core.Reg(t)
        }
        if (next ne null) {
          reg := next
        }
        reg
      } else if (next ne null) {
        if (init ne null) {
          RegNext(next, init)
        } else {
          RegNext(next)
        }
      } else if (init ne null) {
        RegInit(init)
      } else {
        throwException("cannot infer type")
      }
    }
  }

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

  implicit class fromBigIntToLiteral(x: BigInt) extends chisel3.core.fromBigIntToLiteral(x)
  implicit class fromtIntToLiteral(x: Int) extends chisel3.core.fromIntToLiteral(x)
  implicit class fromtLongToLiteral(x: Long) extends chisel3.core.fromLongToLiteral(x)
  implicit class fromStringToLiteral(x: String) extends chisel3.core.fromStringToLiteral(x)
  implicit class fromBooleanToLiteral(x: Boolean) extends chisel3.core.fromBooleanToLiteral(x)
  implicit class fromDoubleToLiteral(x: Double) extends chisel3.core.fromDoubleToLiteral(x)
  implicit class fromIntToWidth(x: Int) extends chisel3.core.fromIntToWidth(x)
  implicit class fromIntToBinaryPoint(x: Int) extends chisel3.core.fromIntToBinaryPoint(x)

  implicit class fromUIntToBitPatComparable(x: UInt) extends chisel3.SourceInfoDoc {
    import scala.language.experimental.macros
    import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

    final def === (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/= (that: BitPat): Bool = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_=== (that: BitPat)  // scalastyle:ignore method.name
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = that === x
    /** @group SourceInfoTransformMacro */
    def do_=/= (that: BitPat)  // scalastyle:ignore method.name
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = that =/= x

    final def != (that: BitPat): Bool = macro SourceInfoTransform.thatArg
    @chiselRuntimeDeprecated
    @deprecated("Use '=/=', which avoids potential precedence problems", "chisel3")
    def do_!= (that: BitPat)  // scalastyle:ignore method.name
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = that != x
  }


  type ChiselException = chisel3.internal.ChiselException

  // Debugger/Tester access to internal Chisel data structures and methods.
  def getDataElements(a: Aggregate): Seq[Element] = {
    a.allElements
  }
  def getModulePorts(m: Module): Seq[Port] = m.getPorts
  // Invalidate API - a DontCare element for explicit assignment to outputs,
  //  indicating the signal is intentionally not driven.
  val DontCare = chisel3.core.DontCare

  /** Package for experimental features, which may have their API changed, be removed, etc.
    *
    * Because its contents won't necessarily have the same level of stability and support as
    * non-experimental, you must explicitly import this package to use its contents.
    */
  object experimental {  // scalastyle:ignore object.name
    type Param = chisel3.core.Param
    type IntParam = chisel3.core.IntParam
    val IntParam = chisel3.core.IntParam
    type DoubleParam = chisel3.core.DoubleParam
    val DoubleParam = chisel3.core.DoubleParam
    type StringParam = chisel3.core.StringParam
    val StringParam = chisel3.core.StringParam
    type RawParam = chisel3.core.RawParam
    val RawParam = chisel3.core.RawParam

    type Analog = chisel3.core.Analog
    val Analog = chisel3.core.Analog
    val attach = chisel3.core.attach

    type ChiselEnum = chisel3.core.EnumFactory
    val EnumAnnotations = chisel3.core.EnumAnnotations

    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withClockAndReset = chisel3.core.withClockAndReset
    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withClock = chisel3.core.withClock
    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withReset = chisel3.core.withReset

    val dontTouch = chisel3.core.dontTouch

    type BaseModule = chisel3.core.BaseModule
    type RawModule = chisel3.core.RawModule
    type MultiIOModule = chisel3.core.MultiIOModule
    type ExtModule = chisel3.core.ExtModule

    val IO = chisel3.core.IO

    // Rocket Chip-style clonemodule

    /** A record containing the results of CloneModuleAsRecord
      * The apply method is retrieves the element with the supplied name.
      */
    type ClonePorts = chisel3.core.BaseModule.ClonePorts

    object CloneModuleAsRecord {
      /** Clones an existing module and returns a record of all its top-level ports.
        * Each element of the record is named with a string matching the
        * corresponding port's name and shares the port's type.
        * @example {{{
        * val q1 = Module(new Queue(UInt(32.W), 2))
        * val q2_io = CloneModuleAsRecord(q1)("io").asInstanceOf[q1.io.type]
        * q2_io.enq <> q1.io.deq
        * }}}
        */
      def apply(proto: BaseModule)(implicit sourceInfo: chisel3.internal.sourceinfo.SourceInfo, compileOptions: chisel3.core.CompileOptions): ClonePorts = {
        chisel3.core.BaseModule.cloneIORecord(proto)
      }
    }

    // Implicit conversions for BlackBox Parameters
    implicit def fromIntToIntParam(x: Int): IntParam = IntParam(BigInt(x))
    implicit def fromLongToIntParam(x: Long): IntParam = IntParam(BigInt(x))
    implicit def fromBigIntToIntParam(x: BigInt): IntParam = IntParam(x)
    implicit def fromDoubleToDoubleParam(x: Double): DoubleParam = DoubleParam(x)
    implicit def fromStringToStringParam(x: String): StringParam = StringParam(x)

    // Fixed Point is experimental for now
    type FixedPoint = chisel3.core.FixedPoint
    val FixedPoint = chisel3.core.FixedPoint

    type ChiselAnnotation = chisel3.core.ChiselAnnotation
    val ChiselAnnotation = chisel3.core.ChiselAnnotation
    type RunFirrtlTransform = chisel3.core.RunFirrtlTransform

    val annotate = chisel3.core.annotate

    val DataMirror = chisel3.core.DataMirror
    val requireIsHardware = chisel3.core.requireIsHardware
    val requireIsChiselType = chisel3.core.requireIsChiselType
    type Direction = chisel3.core.ActualDirection
    val Direction = chisel3.core.ActualDirection

    implicit class ChiselRange(val sc: StringContext) extends AnyVal {
      import scala.language.experimental.macros
      import internal.firrtl.NumericBound

      /** Specifies a range using mathematical range notation. Variables can be interpolated using
        * standard string interpolation syntax.
        * @example {{{
        * UInt(range"[0, 2)")
        * UInt(range"[0, \$myInt)")
        * UInt(range"[0, \${myInt + 2})")
        * }}}
        */
      def range(args: Any*): (NumericBound[Int], NumericBound[Int]) = macro chisel3.internal.RangeTransform.apply
    }

    import scala.annotation.compileTimeOnly

    class dump extends chisel3.internal.naming.dump  // scalastyle:ignore class.name
    class treedump extends chisel3.internal.naming.treedump  // scalastyle:ignore class.name
    class chiselName extends chisel3.internal.naming.chiselName  // scalastyle:ignore class.name
  }
}
