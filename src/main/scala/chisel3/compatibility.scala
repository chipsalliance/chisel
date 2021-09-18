// SPDX-License-Identifier: Apache-2.0

/** The Chisel compatibility package allows legacy users to continue using the `Chisel` (capital C) package name
  *  while moving to the more standard package naming convention `chisel3` (lowercase c).
  */
import chisel3._    // required for implicit conversions.
import chisel3.experimental.chiselName
import chisel3.util.random.FibonacciLFSR
import chisel3.stage.{ChiselCircuitAnnotation, ChiselOutputFileAnnotation, ChiselStage, phases}

package object Chisel {
  import chisel3.internal.firrtl.Width

  import scala.language.experimental.macros
  import scala.annotation.StaticAnnotation
  import scala.annotation.compileTimeOnly
  import scala.language.implicitConversions

  implicit val defaultCompileOptions = chisel3.ExplicitCompileOptions.NotStrict

  abstract class Direction
  case object INPUT extends Direction
  case object OUTPUT extends Direction
  case object NODIR extends Direction

  val Input   = chisel3.Input
  val Output  = chisel3.Output

  object Flipped {
    def apply[T<:Data](target: T): T = chisel3.Flipped[T](target)
  }

  implicit class AddDirectionToData[T<:Data](target: T) {
    def asInput: T = Input(target)
    def asOutput: T = Output(target)
    def flip(): T = Flipped(target)
  }

  implicit class AddDirMethodToData[T<:Data](target: T) {
    import chisel3.ActualDirection
    import chisel3.experimental.DataMirror
    import chisel3.internal.requireIsHardware

    def dir: Direction = {
      requireIsHardware(target) // This has the side effect of calling _autoWrapPorts
      target match {
        case e: Element => DataMirror.directionOf(e) match {
          case ActualDirection.Output => OUTPUT
          case ActualDirection.Input => INPUT
          case _ => NODIR
        }
        case _ => NODIR
      }
    }
  }
  implicit class cloneTypeable[T <: Data](target: T) {
    import chisel3.experimental.DataMirror
    def chiselCloneType: T = {
      DataMirror.internal.chiselTypeClone(target).asInstanceOf[T]
    }
  }

  type ChiselException = chisel3.internal.ChiselException

  type Data = chisel3.Data
  object Wire extends WireFactory {
    import chisel3.CompileOptions

    def apply[T <: Data](dummy: Int = 0, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.WireDefault(init)

    def apply[T <: Data](t: T, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.WireDefault(t, init)
  }
  object Clock {
    def apply(): Clock = new Clock

    def apply(dir: Direction): Clock = {
      val result = apply()
      dir match {
        case INPUT => Input(result)
        case OUTPUT => Output(result)
        case _ => result
      }
    }
  }
  type Clock = chisel3.Clock

  // Implicit conversion to allow fromBits because it's being deprecated in chisel3
  implicit class fromBitsable[T <: Data](data: T) {
    import chisel3.CompileOptions
    import chisel3.internal.sourceinfo.SourceInfo

    /** Creates an new instance of this type, unpacking the input Bits into
      * structured data.
      *
      * This performs the inverse operation of toBits.
      *
      * @note does NOT assign to the object this is called on, instead creates
      * and returns a NEW object (useful in a clone-and-assign scenario)
      * @note does NOT check bit widths, may drop bits during assignment
      * @note what fromBits assigs to must have known widths
      */
    def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      that.asTypeOf(data)
    }
  }

  type Aggregate = chisel3.Aggregate
  object Vec extends chisel3.VecFactory {
    import chisel3.CompileOptions
    import chisel3.internal.sourceinfo._

    @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
    def apply[T <: Data](gen: T, n: Int)(implicit compileOptions: CompileOptions): Vec[T] =
      apply(n, gen)

    /** Creates a new [[Vec]] of length `n` composed of the result of the given
      * function repeatedly applied.
      *
      * @param n   number of elements (and the number of times the function is
      *            called)
      * @param gen function that generates the [[Data]] that becomes the output
      *            element
      */
    def fill[T <: Data](n: Int)(gen: => T)(implicit compileOptions: CompileOptions): Vec[T] =
      apply(Seq.fill(n)(gen))

    def apply[T <: Data](elts: Seq[T]): Vec[T] = macro VecTransform.apply_elts
    /** @group SourceInfoTransformMacro */
    def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.VecInit(elts)

    def apply[T <: Data](elt0: T, elts: T*): Vec[T] = macro VecTransform.apply_elt0
    /** @group SourceInfoTransformMacro */
    def do_apply[T <: Data](elt0: T, elts: T*)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.VecInit(elt0 +: elts.toSeq)

    def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = macro VecTransform.tabulate
    /** @group SourceInfoTransformMacro */
    def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      chisel3.VecInit.tabulate(n)(gen)
  }
  type Vec[T <: Data] = chisel3.Vec[T]
  type VecLike[T <: Data] = chisel3.VecLike[T]
  type Record = chisel3.Record
  type Bundle = chisel3.Bundle

  val assert = chisel3.assert
  val stop = chisel3.stop

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait UIntFactory extends chisel3.UIntFactory {
    /** Create a UInt literal with inferred width. */
    def apply(n: String): UInt = n.asUInt
    /** Create a UInt literal with fixed width. */
    def apply(n: String, width: Int): UInt = n.asUInt(width.W)

    /** Create a UInt literal with specified width. */
    def apply(value: BigInt, width: Width): UInt = value.asUInt(width)

    /** Create a UInt literal with fixed width. */
    def apply(value: BigInt, width: Int): UInt = value.asUInt(width.W)

    /** Create a UInt with a specified width - compatibility with Chisel2. */
    // NOTE: This resolves UInt(width = 32)
    def apply(dir: Option[Direction] = None, width: Int): UInt = apply(width.W)
    /** Create a UInt literal with inferred width.- compatibility with Chisel2. */
    def apply(value: BigInt): UInt = value.asUInt

    /** Create a UInt with a specified direction and width - compatibility with Chisel2. */
    def apply(dir: Direction, width: Int): UInt = apply(dir, width.W)
    /** Create a UInt with a specified direction, but unspecified width - compatibility with Chisel2. */
    def apply(dir: Direction): UInt = apply(dir, Width())
    def apply(dir: Direction, width: Width): UInt = {
      val result = apply(width)
      dir match {
        case INPUT => Input(result)
        case OUTPUT => Output(result)
        case NODIR => result
      }
    }

    /** Create a UInt with a specified width */
    def width(width: Int): UInt = apply(width.W)

    /** Create a UInt port with specified width. */
    def width(width: Width): UInt = apply(width)
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait SIntFactory extends chisel3.SIntFactory {
    /** Create a SInt type or port with fixed width. */
    def width(width: Int): SInt = apply(width.W)
    /** Create an SInt type with specified width. */
    def width(width: Width): SInt = apply(width)

    /** Create an SInt literal with inferred width. */
    def apply(value: BigInt): SInt = value.asSInt
    /** Create an SInt literal with fixed width. */
    def apply(value: BigInt, width: Int): SInt = value.asSInt(width.W)

    /** Create an SInt literal with specified width. */
    def apply(value: BigInt, width: Width): SInt = value.asSInt(width)

    def Lit(value: BigInt): SInt = value.asSInt
    def Lit(value: BigInt, width: Int): SInt = value.asSInt(width.W)

    /** Create a SInt with a specified width - compatibility with Chisel2. */
    def apply(dir: Option[Direction] = None, width: Int): SInt = apply(width.W)
    /** Create a SInt with a specified direction and width - compatibility with Chisel2. */
    def apply(dir: Direction, width: Int): SInt = apply(dir, width.W)
    /** Create a SInt with a specified direction, but unspecified width - compatibility with Chisel2. */
    def apply(dir: Direction): SInt = apply(dir, Width())
    def apply(dir: Direction, width: Width): SInt = {
      val result = apply(width)
      dir match {
        case INPUT => Input(result)
        case OUTPUT => Output(result)
        case NODIR => result
      }
    }
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait BoolFactory extends chisel3.BoolFactory {
    /** Creates Bool literal.
      */
    def apply(x: Boolean): Bool = x.B

    /** Create a UInt with a specified direction and width - compatibility with Chisel2. */
    def apply(dir: Direction): Bool = {
      val result = apply()
      dir match {
        case INPUT => Input(result)
        case OUTPUT => Output(result)
        case NODIR => result
      }
    }
  }

  type Element = chisel3.Element
  type Bits = chisel3.Bits
  object Bits extends UIntFactory
  type Num[T <: Data] = chisel3.Num[T]
  type UInt = chisel3.UInt
  object UInt extends UIntFactory
  type SInt = chisel3.SInt
  object SInt extends SIntFactory
  type Bool = chisel3.Bool
  object Bool extends BoolFactory
  val Mux = chisel3.Mux
  type Reset = chisel3.Reset

  implicit def resetToBool(reset: Reset): Bool = reset.asBool

  type BlackBox = chisel3.internal.LegacyBlackBox

  type MemBase[T <: Data] = chisel3.MemBase[T]

  val Mem = chisel3.Mem
  type Mem[T <: Data] = chisel3.Mem[T]

  implicit class MemCompatibility(a: Mem.type) {
    import chisel3.internal.sourceinfo.UnlocatableSourceInfo

    def apply[T <: Data](t: T, size: BigInt)(implicit compileOptions: CompileOptions): Mem[T] =
      a.do_apply(size, t)(UnlocatableSourceInfo, compileOptions)

    def apply[T <: Data](t: T, size: Int)(implicit compileOptions: CompileOptions): Mem[T] =
      a.do_apply(size, t)(UnlocatableSourceInfo, compileOptions)

  }

  val SeqMem = chisel3.SyncReadMem
  type SeqMem[T <: Data] = chisel3.SyncReadMem[T]

  implicit class SeqMemCompatibility(a: SeqMem.type) {
    import chisel3.internal.sourceinfo.SourceInfo

    def apply[T <: Data](t: T, size: BigInt)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SyncReadMem[T] =
      a.do_apply(size, t)

    def apply[T <: Data](t: T, size: Int)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): SyncReadMem[T] =
      a.do_apply(size, t)
  }

  import chisel3.CompileOptions

  @deprecated("Use Chisel.Module", "Chisel 3.5")
  type CompatibilityModule = chisel3.internal.LegacyModule
  val Module = chisel3.Module
  type Module = chisel3.internal.LegacyModule

  val printf = chisel3.printf

  val RegNext = chisel3.RegNext
  val RegInit = chisel3.RegInit
  object Reg {
    import chisel3.CompileOptions
    import chisel3.internal.sourceinfo.SourceInfo

    // Passthrough for chisel3.Reg
    // Single-element constructor to avoid issues caused by null default args in a type
    // parameterized scope.
    def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      chisel3.Reg(t)

    /** Creates a register with optional next and initialization values.
      *
      * @param t: data type for the register
      * @param next: new value register is to be updated with every cycle (or
      * empty to not update unless assigned to using the := operator)
      * @param init: initialization value on reset (or empty for uninitialized,
      * where the register value persists across a reset)
      *
      * @note this may result in a type error if called from a type parameterized
      * function, since the Scala compiler isn't smart enough to know that null
      * is a valid value. In those cases, you can either use the outType only Reg
      * constructor or pass in `null.asInstanceOf[T]`.
      */
    def apply[T <: Data](t: T = null, next: T = null, init: T = null)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      if (t ne null) {
        val reg = if (init ne null) {
          RegInit(t, init)
        } else {
          chisel3.Reg(t)
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

  val when = chisel3.when
  type WhenContext = chisel3.WhenContext

  implicit class fromBigIntToLiteral(x: BigInt) extends chisel3.fromBigIntToLiteral(x)
  implicit class fromtIntToLiteral(x: Int) extends chisel3.fromIntToLiteral(x)
  implicit class fromtLongToLiteral(x: Long) extends chisel3.fromLongToLiteral(x)
  implicit class fromStringToLiteral(x: String) extends chisel3.fromStringToLiteral(x)
  implicit class fromBooleanToLiteral(x: Boolean) extends chisel3.fromBooleanToLiteral(x)
  implicit class fromIntToWidth(x: Int) extends chisel3.fromIntToWidth(x)

  type BackendCompilationUtilities = firrtl.util.BackendCompilationUtilities
  val ImplicitConversions = chisel3.util.ImplicitConversions

  // Deprecated as of Chisel3
  object chiselMain {
    import java.io.File

    private var target_dir: Option[String] = None

    private def parseArgs(args: Array[String]): Unit = {
      for (i <- args.indices) {
        if (args(i) == "--targetDir") {
          target_dir = Some(args(i + 1))
        }
      }
    }

    def apply[T <: Module](args: Array[String], gen: () => T): Unit =
      Predef.assert(false, "No more chiselMain in Chisel3")

    def run[T <: Module] (args: Array[String], gen: () => T): Unit = {
      val circuit = ChiselStage.elaborate(gen())
      parseArgs(args)
      val output_file = new File(target_dir.getOrElse(new File(".").getCanonicalPath) + "/" + circuit.name + ".fir")

      (new phases.Emitter).transform(Seq(ChiselCircuitAnnotation(circuit),
                                         ChiselOutputFileAnnotation(output_file.toString)))
    }
  }

  @deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
  object debug {
    def apply (arg: Data): Data = arg
  }

  // Deprecated as of Chsiel3
  object throwException {
    @throws(classOf[Exception])
    def apply(s: String, t: Throwable = null): Nothing = {
      val xcpt = new Exception(s, t)
      throw xcpt
    }
  }

  object testers {
    type BasicTester = chisel3.testers.BasicTester
    val TesterDriver = chisel3.testers.TesterDriver
  }

  val log2Ceil = chisel3.util.log2Ceil
  val log2Floor = chisel3.util.log2Floor
  val isPow2 = chisel3.util.isPow2

  /** Compute the log2 rounded up with min value of 1 */
  object log2Up {
    def apply(in: BigInt): Int = {
      require(in >= 0)
      1 max (in-1).bitLength
    }
    def apply(in: Int): Int = apply(BigInt(in))
  }

  /** Compute the log2 rounded down with min value of 1 */
  object log2Down {
    def apply(in: BigInt): Int = log2Up(in) - (if (isPow2(in)) 0 else 1)
    def apply(in: Int): Int = apply(BigInt(in))
  }

  val BitPat = chisel3.util.BitPat
  type BitPat = chisel3.util.BitPat

  implicit class BitsObjectCompatibility(a: BitPat.type) {
    def DC(width: Int): BitPat = a.dontCare(width)
  }

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

  type SwitchContext[T <: Bits] = chisel3.util.SwitchContext[T]
  val is = chisel3.util.is
  val switch = chisel3.util.switch

  type Counter = chisel3.util.Counter
  val Counter = chisel3.util.Counter

  type DecoupledIO[+T <: Data] = chisel3.util.DecoupledIO[T]
  val DecoupledIO = chisel3.util.Decoupled
  val Decoupled = chisel3.util.Decoupled
  type QueueIO[T <: Data] = chisel3.util.QueueIO[T]

  val Queue = chisel3.util.Queue
  type Queue[T <: Data] = QueueCompatibility[T]

  sealed class QueueCompatibility[T <: Data](gen: T, entries: Int, pipe: Boolean = false, flow: Boolean = false)
                                 (implicit compileOptions: chisel3.CompileOptions)
      extends chisel3.util.Queue[T](gen, entries, pipe, flow)(compileOptions) {

    def this(gen: T, entries: Int, pipe: Boolean, flow: Boolean, override_reset: Option[Bool]) = {
      this(gen, entries, pipe, flow)
      this.override_reset = override_reset
    }

    def this(gen: T, entries: Int, pipe: Boolean, flow: Boolean, _reset: Bool) = {
      this(gen, entries, pipe, flow)
      this.override_reset = Some(_reset)
    }

  }

  object Enum extends chisel3.util.Enum {
    /** Returns n unique values of the specified type. Can be used with unpacking to define enums.
      *
      * nodeType must be of UInt type (note that Bits() creates a UInt) with unspecified width.
      *
      * @example {{{
      * val state_on :: state_off :: Nil = Enum(UInt(), 2)
      * val current_state = UInt()
      * switch (current_state) {
      *   is (state_on) {
      *      ...
      *   }
      *   if (state_off) {
      *      ...
      *   }
      * }
      * }}}
      */
    def apply[T <: Bits](nodeType: T, n: Int): List[T] = {
      require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
      require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
      apply(n).asInstanceOf[List[T]]
    }

    /** An old Enum API that returns a map of symbols to UInts.
      *
      * Unlike the new list-based Enum, which can be unpacked into vals that the compiler
      * understands and can check, map accesses can't be compile-time checked and typos may not be
      * caught until runtime.
      *
      * Despite being deprecated, this is not to be removed from the compatibility layer API.
      * Deprecation is only to nag users to do something safer.
      */
    @deprecated("Use list-based Enum", "not soon enough")
    def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = {
      require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
      require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
      (l zip createValues(l.length)).toMap.asInstanceOf[Map[Symbol, T]]
    }

    /** An old Enum API that returns a map of symbols to UInts.
      *
      * Unlike the new list-based Enum, which can be unpacked into vals that the compiler
      * understands and can check, map accesses can't be compile-time checked and typos may not be
      * caught until runtime.
      *
      * Despite being deprecated, this is not to be removed from the compatibility layer API.
      * Deprecation is only to nag users to do something safer.
      */
    @deprecated("Use list-based Enum", "not soon enough")
    def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = {
      require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
      require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
      (l zip createValues(l.length)).toMap.asInstanceOf[Map[Symbol, T]]
    }
  }

  /** LFSR16 generates a 16-bit linear feedback shift register, returning the register contents.
    * This is useful for generating a pseudo-random sequence.
    *
    * The example below, taken from the unit tests, creates two 4-sided dice using `LFSR16` primitives:
    * @example {{{
    *   val bins = Reg(Vec(8, UInt(32.W)))
    *
    *   // Create two 4 sided dice and roll them each cycle.
    *   // Use tap points on each LFSR so values are more independent
    *   val die0 = Cat(Seq.tabulate(2) { i => LFSR16()(i) })
    *   val die1 = Cat(Seq.tabulate(2) { i => LFSR16()(i + 2) })
    *
    *   val rollValue = die0 +& die1  // Note +& is critical because sum will need an extra bit.
    *
    *   bins(rollValue) := bins(rollValue) + 1.U
    *
    * }}}
    */
  object LFSR16 {
    /** Generates a 16-bit linear feedback shift register, returning the register contents.
      * @param increment optional control to gate when the LFSR updates.
      */
    @chiselName
    def apply(increment: Bool = true.B): UInt =
      VecInit( FibonacciLFSR
                .maxPeriod(16, increment, seed = Some(BigInt(1) << 15))
                .asBools
                .reverse )
        .asUInt

  }

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

  val RegEnable = chisel3.util.RegEnable
  val ShiftRegister = chisel3.util.ShiftRegister

  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  val Valid = chisel3.util.Valid
  val Pipe = chisel3.util.Pipe
  type Pipe[T <: Data] = chisel3.util.Pipe[T]


  /** Package for experimental features, which may have their API changed, be removed, etc.
    *
    * Because its contents won't necessarily have the same level of stability and support as
    * non-experimental, you must explicitly import this package to use its contents.
    */
  object experimental {
    import scala.annotation.compileTimeOnly

    class dump extends chisel3.internal.naming.dump
    class treedump extends chisel3.internal.naming.treedump
    class chiselName extends chisel3.internal.naming.chiselName
  }

  implicit class DataCompatibility(a: Data) {
    import chisel3.internal.sourceinfo.DeprecatedSourceInfo

    def toBits(implicit compileOptions: CompileOptions): UInt = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

  }

  implicit class VecLikeCompatibility[T <: Data](a: VecLike[T]) {
    import chisel3.internal.sourceinfo.DeprecatedSourceInfo

    def read(idx: UInt)(implicit compileOptions: CompileOptions): T = a.do_apply(idx)(compileOptions)

    def write(idx: UInt, data: T)(implicit compileOptions: CompileOptions): Unit =
      a.do_apply(idx)(compileOptions).:=(data)(DeprecatedSourceInfo, compileOptions)

  }

  implicit class BitsCompatibility(a: Bits) {
    import chisel3.internal.sourceinfo.DeprecatedSourceInfo

    final def asBits(implicit compileOptions: CompileOptions): Bits = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

    final def toSInt(implicit compileOptions: CompileOptions): SInt = a.do_asSInt(DeprecatedSourceInfo, compileOptions)

    final def toUInt(implicit compileOptions: CompileOptions): UInt = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

  }

}
