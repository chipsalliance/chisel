// See LICENSE for license details.

/** The Chisel compatibility package allows legacy users to continue using the `Chisel` (capital C) package name
  *  while moving to the more standard package naming convention `chisel3` (lowercase c).
  */

package object Chisel {     // scalastyle:ignore package.object.name
  import chisel3.internal.firrtl.Width

  import scala.language.experimental.macros
  import scala.annotation.StaticAnnotation
  import scala.annotation.compileTimeOnly

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
  type Record = chisel3.core.Record
  type Bundle = chisel3.core.Bundle

  val assert = chisel3.core.assert
  val stop = chisel3.core.stop

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait UIntFactory extends chisel3.core.UIntFactory {
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
        case chisel3.core.Direction.Input => chisel3.core.Input(result)
        case chisel3.core.Direction.Output => chisel3.core.Output(result)
        case chisel3.core.Direction.Unspecified => result
      }
    }

    /** Create a UInt with a specified width */
    def width(width: Int): UInt = apply(width.W)

    /** Create a UInt port with specified width. */
    def width(width: Width): UInt = apply(width)
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait SIntFactory extends chisel3.core.SIntFactory {
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
        case chisel3.core.Direction.Input => chisel3.core.Input(result)
        case chisel3.core.Direction.Output => chisel3.core.Output(result)
        case chisel3.core.Direction.Unspecified => result
      }
    }
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  trait BoolFactory extends chisel3.core.BoolFactory {
    /** Creates Bool literal.
      */
    def apply(x: Boolean): Bool = x.B

    /** Create a UInt with a specified direction and width - compatibility with Chisel2. */
    def apply(dir: Direction): Bool = {
      val result = apply()
      dir match {
        case chisel3.core.Direction.Input => chisel3.core.Input(result)
        case chisel3.core.Direction.Output => chisel3.core.Output(result)
        case chisel3.core.Direction.Unspecified => result
      }
    }
  }

  type Element = chisel3.core.Element
  type Bits = chisel3.core.Bits
  object Bits extends UIntFactory
  type Num[T <: Data] = chisel3.core.Num[T]
  type UInt = chisel3.core.UInt
  object UInt extends UIntFactory
  type SInt = chisel3.core.SInt
  object SInt extends SIntFactory
  type Bool = chisel3.core.Bool
  object Bool extends BoolFactory
  val Mux = chisel3.core.Mux

  import chisel3.core.Param
  abstract class BlackBox(params: Map[String, Param] = Map.empty[String, Param]) extends chisel3.core.BlackBox(params) {
    // This class auto-wraps the BlackBox with IO(...), allowing legacy code (where IO(...) wasn't
    // required) to build. 
    override def _autoWrapPorts() = {
      if (!_ioPortBound()) {
        IO(io)
      }
    }
  }
  val Mem = chisel3.core.Mem
  type MemBase[T <: Data] = chisel3.core.MemBase[T]
  type Mem[T <: Data] = chisel3.core.Mem[T]
  val SeqMem = chisel3.core.SyncReadMem
  type SeqMem[T <: Data] = chisel3.core.SyncReadMem[T]

  import chisel3.core.CompileOptions
  abstract class CompatibilityModule(
      override_clock: Option[Clock]=None, override_reset: Option[Bool]=None)
      (implicit moduleCompileOptions: CompileOptions)
      extends chisel3.core.LegacyModule(override_clock, override_reset) {
    // This class auto-wraps the Module IO with IO(...), allowing legacy code (where IO(...) wasn't
    // required) to build.
    // Also provides the clock / reset constructors, which were used before withClock happened.
    
    def this(_clock: Clock)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), None)(moduleCompileOptions)
    def this(_reset: Bool)(implicit moduleCompileOptions: CompileOptions)  = this(None, Option(_reset))(moduleCompileOptions)
    def this(_clock: Clock, _reset: Bool)(implicit moduleCompileOptions: CompileOptions) = this(Option(_clock), Option(_reset))(moduleCompileOptions)

    override def _autoWrapPorts() = {
      if (!_ioPortBound()) {
        IO(io)
      }
    }
  }
  val Module = chisel3.core.Module
  type Module = CompatibilityModule

  val printf = chisel3.core.printf

  val RegNext = chisel3.core.RegNext
  val RegInit = chisel3.core.RegInit
  object Reg {
    import chisel3.core.{Binding, CompileOptions}
    import chisel3.internal.sourceinfo.SourceInfo

    // Passthrough for chisel3.core.Reg
    // Single-element constructor to avoid issues caused by null default args in a type
    // parameterized scope.
    def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      chisel3.core.Reg(t)
    
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
          chisel3.core.Reg(t)
        }
        if (next ne null) {
          Binding.checkSynthesizable(next, s"'next' ($next)")  // TODO: move into connect?
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

  implicit class fromBigIntToLiteral(val x: BigInt) extends chisel3.core.fromBigIntToLiteral(x)
  implicit class fromtIntToLiteral(val x: Int) extends chisel3.core.fromIntToLiteral(x)
  implicit class fromtLongToLiteral(val x: Long) extends chisel3.core.fromLongToLiteral(x)
  implicit class fromStringToLiteral(val x: String) extends chisel3.core.fromStringToLiteral(x)
  implicit class fromBooleanToLiteral(val x: Boolean) extends chisel3.core.fromBooleanToLiteral(x)
  implicit class fromIntToWidth(val x: Int) extends chisel3.core.fromIntToWidth(x)

  type BackendCompilationUtilities = firrtl.util.BackendCompilationUtilities
  val Driver = chisel3.Driver
  val ImplicitConversions = chisel3.util.ImplicitConversions

  // Deprecated as of Chisel3
  object chiselMain {
    import java.io.File

    def apply[T <: Module](args: Array[String], gen: () => T): Unit =
      Predef.assert(false, "No more chiselMain in Chisel3")

    def run[T <: Module] (args: Array[String], gen: () => T): Unit = {
      val circuit = Driver.elaborate(gen)
      Driver.parseArgs(args)
      val output_file = new File(Driver.targetDir + "/" + circuit.name + ".fir")
      Driver.dumpFirrtl(circuit, Option(output_file))
    }
  }

  @deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
  object debug {  // scalastyle:ignore object.name
    def apply (arg: Data): Data = arg
  }

  // Deprecated as of Chsiel3
  @throws(classOf[Exception])
  object throwException {
    def apply(s: String, t: Throwable = null) = {
      val xcpt = new Exception(s, t)
      throw xcpt
    }
  }

  object testers {    // scalastyle:ignore object.name
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
}
