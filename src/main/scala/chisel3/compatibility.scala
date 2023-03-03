// SPDX-License-Identifier: Apache-2.0

/** The Chisel compatibility package allows legacy users to continue using the `Chisel` (capital C) package name
  *  while moving to the more standard package naming convention `chisel3` (lowercase c).
  */
import chisel3._ // required for implicit conversions.
import chisel3.util.random.FibonacciLFSR
import chisel3.stage.{phases, ChiselCircuitAnnotation, ChiselOutputFileAnnotation}

import circt.stage.ChiselStage

import scala.annotation.nowarn

package object Chisel {
  import chisel3.internal.firrtl.Width

  import scala.language.experimental.macros
  import scala.annotation.StaticAnnotation
  import scala.annotation.compileTimeOnly
  import scala.language.implicitConversions

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  implicit val defaultCompileOptions = chisel3.ExplicitCompileOptions.NotStrict

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  abstract class Direction
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  case object INPUT extends Direction
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  case object OUTPUT extends Direction
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  case object NODIR extends Direction

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Input = chisel3.Input
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Output = chisel3.Output

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Flipped {
    def apply[T <: Data](target: T): T = chisel3.Flipped[T](target)
  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class AddDirectionToData[T <: Data](target: T) {
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def asInput: T = Input(target)
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def asOutput: T = Output(target)
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def flip: T = Flipped(target)
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    @deprecated(
      "Calling this function with an empty argument list is invalid in Scala 3. Use the form without parentheses instead",
      "Chisel 3.5"
    )
    def flip(dummy: Int*): T = flip
  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class AddDirMethodToData[T <: Data](target: T) {
    import chisel3.ActualDirection
    import chisel3.reflect.DataMirror
    import chisel3.internal.requireIsHardware

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def dir: Direction = {
      requireIsHardware(target) // This has the side effect of calling _autoWrapPorts
      target match {
        case e: Element =>
          DataMirror.directionOf(e) match {
            case ActualDirection.Output => OUTPUT
            case ActualDirection.Input  => INPUT
            case _                      => NODIR
          }
        case _ => NODIR
      }
    }
  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class cloneTypeable[T <: Data](target: T) {
    import chisel3.reflect.DataMirror
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def chiselCloneType: T = {
      DataMirror.internal.chiselTypeClone(target).asInstanceOf[T]
    }
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type ChiselException = chisel3.ChiselException

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Data = chisel3.Data
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Wire extends WireFactory {
    import chisel3.CompileOptions

    def apply[T <: Data](dummy: Int = 0, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.WireDefault(init)

    def apply[T <: Data](t: T, init: T)(implicit compileOptions: CompileOptions): T =
      chisel3.WireDefault(t, init)
  }
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Clock {
    def apply(): Clock = new Clock

    def apply(dir: Direction): Clock = {
      val result = apply()
      dir match {
        case INPUT  => Input(result)
        case OUTPUT => Output(result)
        case _      => result
      }
    }
  }
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Clock = chisel3.Clock

  // Implicit conversion to allow fromBits because it's being deprecated in chisel3
  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class fromBitsable[T <: Data](data: T) {
    import chisel3.CompileOptions
    import chisel3.experimental.SourceInfo

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
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      that.asTypeOf(data)
    }
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Aggregate = chisel3.Aggregate
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Vec extends chisel3.VecFactory {
    import chisel3.CompileOptions
    import chisel3.experimental._
    import chisel3.internal.sourceinfo.VecTransform

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
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
    def do_apply[T <: Data](
      elt0: T,
      elts: T*
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): Vec[T] =
      chisel3.VecInit(elt0 +: elts.toSeq)

    def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = macro VecTransform.tabulate

    /** @group SourceInfoTransformMacro */
    def do_tabulate[T <: Data](
      n:   Int
    )(gen: (Int) => T
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): Vec[T] =
      chisel3.VecInit.tabulate(n)(gen)
  }
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Vec[T <: Data] = chisel3.Vec[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type VecLike[T <: Data] = chisel3.VecLike[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Record = chisel3.Record
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Bundle = chisel3.Bundle

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val assert = chisel3.assert
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val stop = chisel3.stop

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  trait UIntFactory extends chisel3.UIntFactory {

    /** Create a UInt literal with inferred width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(n: String): UInt = n.asUInt

    /** Create a UInt literal with fixed width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(n: String, width: Int): UInt = n.asUInt(width.W)

    /** Create a UInt literal with specified width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt, width: Width): UInt = value.asUInt(width)

    /** Create a UInt literal with fixed width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt, width: Int): UInt = value.asUInt(width.W)

    /** Create a UInt with a specified width - compatibility with Chisel2. */
    // NOTE: This resolves UInt(width = 32)
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Option[Direction] = None, width: Int): UInt = apply(width.W)

    /** Create a UInt literal with inferred width.- compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt): UInt = value.asUInt

    /** Create a UInt with a specified direction and width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction, width: Int): UInt = apply(dir, width.W)

    /** Create a UInt with a specified direction, but unspecified width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction): UInt = apply(dir, Width())
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction, width: Width): UInt = {
      val result = apply(width)
      dir match {
        case INPUT  => Input(result)
        case OUTPUT => Output(result)
        case NODIR  => result
      }
    }

    /** Create a UInt with a specified width */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def width(width: Int): UInt = apply(width.W)

    /** Create a UInt port with specified width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def width(width: Width): UInt = apply(width)
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  trait SIntFactory extends chisel3.SIntFactory {

    /** Create a SInt type or port with fixed width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def width(width: Int): SInt = apply(width.W)

    /** Create an SInt type with specified width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def width(width: Width): SInt = apply(width)

    /** Create an SInt literal with inferred width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt): SInt = value.asSInt

    /** Create an SInt literal with fixed width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt, width: Int): SInt = value.asSInt(width.W)

    /** Create an SInt literal with specified width. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(value: BigInt, width: Width): SInt = value.asSInt(width)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def Lit(value: BigInt): SInt = value.asSInt

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def Lit(value: BigInt, width: Int): SInt = value.asSInt(width.W)

    /** Create a SInt with a specified width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Option[Direction] = None, width: Int): SInt = apply(width.W)

    /** Create a SInt with a specified direction and width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction, width: Int): SInt = apply(dir, width.W)

    /** Create a SInt with a specified direction, but unspecified width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction): SInt = apply(dir, Width())
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction, width: Width): SInt = {
      val result = apply(width)
      dir match {
        case INPUT  => Input(result)
        case OUTPUT => Output(result)
        case NODIR  => result
      }
    }
  }

  /** This contains literal constructor factory methods that are deprecated as of Chisel3.
    */
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  trait BoolFactory extends chisel3.BoolFactory {

    /** Creates Bool literal.
      */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(x: Boolean): Bool = x.B

    /** Create a UInt with a specified direction and width - compatibility with Chisel2. */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(dir: Direction): Bool = {
      val result = apply()
      dir match {
        case INPUT  => Input(result)
        case OUTPUT => Output(result)
        case NODIR  => result
      }
    }
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Element = chisel3.Element
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Bits = chisel3.Bits
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Bits extends UIntFactory
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Num[T <: Data] = chisel3.Num[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type UInt = chisel3.UInt
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object UInt extends UIntFactory
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type SInt = chisel3.SInt
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object SInt extends SIntFactory
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Bool = chisel3.Bool
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object Bool extends BoolFactory
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Mux = chisel3.Mux
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Reset = chisel3.Reset

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  implicit def resetToBool(reset: Reset): Bool = reset.asBool

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type MemBase[T <: Data] = chisel3.MemBase[T]

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Mem = chisel3.Mem
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Mem[T <: Data] = chisel3.Mem[T]

  implicit class MemCompatibility(a: Mem.type) {
    import chisel3.experimental.UnlocatableSourceInfo

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Data](t: T, size: BigInt)(implicit compileOptions: CompileOptions): Mem[T] =
      a.do_apply(size, t)(UnlocatableSourceInfo, compileOptions)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Data](t: T, size: Int)(implicit compileOptions: CompileOptions): Mem[T] =
      a.do_apply(size, t)(UnlocatableSourceInfo, compileOptions)

  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val SeqMem = chisel3.SyncReadMem
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type SeqMem[T <: Data] = chisel3.SyncReadMem[T]

  implicit class SeqMemCompatibility(a: SeqMem.type) {
    import chisel3.experimental.SourceInfo

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Data](
      t:    T,
      size: BigInt
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): SyncReadMem[T] =
      a.do_apply(size, t)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Data](
      t:    T,
      size: Int
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): SyncReadMem[T] =
      a.do_apply(size, t)
  }

  import chisel3.CompileOptions

  @deprecated("Use Chisel.Module", "Chisel 3.5")
  type CompatibilityModule = chisel3.internal.LegacyModule
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Module = chisel3.Module
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Module = chisel3.internal.LegacyModule

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val printf = chisel3.printf

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val RegNext = chisel3.RegNext
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val RegInit = chisel3.RegInit

  @nowarn("msg=Chisel compatibility mode is deprecated")
  object Reg {
    import chisel3.CompileOptions
    import chisel3.experimental.SourceInfo

    // Passthrough for chisel3.Reg
    // Single-element constructor to avoid issues caused by null default args in a type
    // parameterized scope.
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
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
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Data](
      t:    T = null,
      next: T = null,
      init: T = null
    )(
      implicit sourceInfo: SourceInfo,
      compileOptions:      CompileOptions
    ): T = {
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

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val when = chisel3.when
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type WhenContext = chisel3.WhenContext

  implicit class fromBigIntToLiteral(x: BigInt) extends chisel3.fromBigIntToLiteral(x)
  implicit class fromtIntToLiteral(x: Int) extends chisel3.fromIntToLiteral(x)
  implicit class fromtLongToLiteral(x: Long) extends chisel3.fromLongToLiteral(x)
  implicit class fromStringToLiteral(x: String) extends chisel3.fromStringToLiteral(x)
  implicit class fromBooleanToLiteral(x: Boolean) extends chisel3.fromBooleanToLiteral(x)
  implicit class fromIntToWidth(x: Int) extends chisel3.fromIntToWidth(x)

  @deprecated("Use object firrtl.util.BackendCompilationUtilities instead", "Chisel 3.5")
  type BackendCompilationUtilities = chisel3.BackendCompilationUtilities
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
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

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply[T <: Module](args: Array[String], gen: () => T): Unit =
      Predef.assert(false, "No more chiselMain in Chisel3")

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def run[T <: Module](args: Array[String], gen: () => T): Unit = {
      val circuit = ChiselStage.elaborate(gen())
      parseArgs(args)
      val output_file = new File(target_dir.getOrElse(new File(".").getCanonicalPath) + "/" + circuit.name + ".fir")

      (new phases.Emitter)
        .transform(Seq(ChiselCircuitAnnotation(circuit), ChiselOutputFileAnnotation(output_file.toString)))
    }
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  @deprecated("debug doesn't do anything in Chisel3 as no pruning happens in the frontend", "chisel3")
  object debug {
    def apply(arg: Data): Data = arg
  }

  // Deprecated as of Chisel3
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object throwException {
    @throws(classOf[Exception])
    def apply(s: String, t: Throwable = null): Nothing = {
      val xcpt = new Exception(s, t)
      throw xcpt
    }
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  object testers {
    type BasicTester = chisel3.testers.BasicTester
    val TesterDriver = chisel3.testers.TesterDriver
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val log2Ceil = chisel3.util.log2Ceil
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val log2Floor = chisel3.util.log2Floor
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val isPow2 = chisel3.util.isPow2

  /** Compute the log2 rounded up with min value of 1 */
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val log2Up = chisel3.util.log2Up

  /** Compute the log2 rounded down with min value of 1 */
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val log2Down = chisel3.util.log2Down

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val BitPat = chisel3.util.BitPat
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type BitPat = chisel3.util.BitPat

  implicit class BitsObjectCompatibility(a: BitPat.type) {
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def DC(width: Int): BitPat = a.dontCare(width)
  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type ArbiterIO[T <: Data] = chisel3.util.ArbiterIO[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type LockingArbiterLike[T <: Data] = chisel3.util.LockingArbiterLike[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type LockingRRArbiter[T <: Data] = chisel3.util.LockingRRArbiter[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type LockingArbiter[T <: Data] = chisel3.util.LockingArbiter[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type RRArbiter[T <: Data] = chisel3.util.RRArbiter[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Arbiter[T <: Data] = chisel3.util.Arbiter[T]

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val FillInterleaved = chisel3.util.FillInterleaved
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val PopCount = chisel3.util.PopCount
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Fill = chisel3.util.Fill
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Reverse = chisel3.util.Reverse

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Cat = chisel3.Cat

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Log2 = chisel3.util.Log2

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type SwitchContext[T <: Bits] = chisel3.util.SwitchContext[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val is = chisel3.util.is
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val switch = chisel3.util.switch

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Counter = chisel3.util.Counter
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Counter = chisel3.util.Counter

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type DecoupledIO[+T <: Data] = chisel3.util.DecoupledIO[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val DecoupledIO = chisel3.util.Decoupled
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Decoupled = chisel3.util.Decoupled
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type QueueIO[T <: Data] = chisel3.util.QueueIO[T]

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Queue = chisel3.util.Queue
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Queue[T <: Data] = QueueCompatibility[T]

  @nowarn("msg=Chisel compatibility mode is deprecated")
  sealed class QueueCompatibility[T <: Data](
    gen:     T,
    entries: Int,
    pipe:    Boolean = false,
    flow:    Boolean = false
  )(
    implicit compileOptions: chisel3.CompileOptions)
      extends chisel3.util.Queue[T](gen, entries, pipe, flow)(compileOptions) {

    @nowarn("msg=method override_reset_= in class Module is deprecated")
    def this(gen: T, entries: Int, pipe: Boolean, flow: Boolean, override_reset: Option[Bool]) = {
      this(gen, entries, pipe, flow)
      this.override_reset = override_reset // TODO: Find a better way to do this
    }

    @nowarn("msg=method override_reset_= in class Module is deprecated")
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def this(gen: T, entries: Int, pipe: Boolean, flow: Boolean, _reset: Bool) = {
      this(gen, entries, pipe, flow)
      this.override_reset = Some(_reset) // TODO: Find a better way to do this
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
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
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
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    @deprecated("Use list-based Enum", "not soon enough")
    def apply[T <: Bits](nodeType: T, l: Symbol*): Map[Symbol, T] = {
      require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
      require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
      (l.zip(createValues(l.length))).toMap.asInstanceOf[Map[Symbol, T]]
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
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    @deprecated("Use list-based Enum", "not soon enough")
    def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = {
      require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
      require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
      (l.zip(createValues(l.length))).toMap.asInstanceOf[Map[Symbol, T]]
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
  @nowarn("msg=Chisel compatibility mode is deprecated")
  object LFSR16 {

    /** Generates a 16-bit linear feedback shift register, returning the register contents.
      * @param increment optional control to gate when the LFSR updates.
      */
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def apply(increment: Bool = true.B): UInt =
      VecInit(
        FibonacciLFSR
          .maxPeriod(16, increment, seed = Some(BigInt(1) << 15))
          .asBools
          .reverse
      ).asUInt

  }

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val ListLookup = chisel3.util.ListLookup
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Lookup = chisel3.util.Lookup

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Mux1H = chisel3.util.Mux1H
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val PriorityMux = chisel3.util.PriorityMux
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val MuxLookup = chisel3.util.MuxLookup
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val MuxCase = chisel3.util.MuxCase

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val OHToUInt = chisel3.util.OHToUInt
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val PriorityEncoder = chisel3.util.PriorityEncoder
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val UIntToOH = chisel3.util.UIntToOH
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val PriorityEncoderOH = chisel3.util.PriorityEncoderOH

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val RegEnable = chisel3.util.RegEnable
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val ShiftRegister = chisel3.util.ShiftRegister

  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type ValidIO[+T <: Data] = chisel3.util.Valid[T]
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Valid = chisel3.util.Valid
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  val Pipe = chisel3.util.Pipe
  @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
  type Pipe[T <: Data] = chisel3.util.Pipe[T]

  /** Package for experimental features, which may have their API changed, be removed, etc.
    *
    * Because its contents won't necessarily have the same level of stability and support as
    * non-experimental, you must explicitly import this package to use its contents.
    */
  object experimental {
    import scala.annotation.compileTimeOnly

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    class dump extends chisel3.internal.naming.dump
    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    class treedump extends chisel3.internal.naming.treedump
  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class DataCompatibility(a: Data) {
    import chisel3.experimental.DeprecatedSourceInfo

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def toBits(implicit compileOptions: CompileOptions): UInt = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class VecLikeCompatibility[T <: Data](a: VecLike[T]) {
    import chisel3.experimental.DeprecatedSourceInfo

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def read(idx: UInt)(implicit compileOptions: CompileOptions): T = a.do_apply(idx)(compileOptions)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    def write(idx: UInt, data: T)(implicit compileOptions: CompileOptions): Unit =
      a.do_apply(idx)(compileOptions).:=(data)(DeprecatedSourceInfo, compileOptions)

  }

  @nowarn("msg=Chisel compatibility mode is deprecated")
  implicit class BitsCompatibility(a: Bits) {
    import chisel3.experimental.DeprecatedSourceInfo

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    final def asBits(implicit compileOptions: CompileOptions): Bits = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    final def toSInt(implicit compileOptions: CompileOptions): SInt = a.do_asSInt(DeprecatedSourceInfo, compileOptions)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    final def toUInt(implicit compileOptions: CompileOptions): UInt = a.do_asUInt(DeprecatedSourceInfo, compileOptions)

    @deprecated("Chisel compatibility mode is deprecated. Use the chisel3 package instead.", "Chisel 3.6")
    final def toBools(implicit compileOptions: CompileOptions): Seq[Bool] =
      a.do_asBools(DeprecatedSourceInfo, compileOptions)
  }

}
