// See LICENSE for license details.

/** This package contains the main chisel3 API.
 */
package object chisel3 {    // scalastyle:ignore package.object.name
  import internal.firrtl.{Port, Width}
  import internal.sourceinfo.{SourceInfo, VecTransform}
  import internal.{Builder, chiselRuntimeDeprecated}

  import scala.language.implicitConversions

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
        *
        * Implementation note: the empty parameter list (like `U()`) is necessary to prevent
        * interpreting calls that have a non-Width parameter as a chained apply, otherwise things like
        * `0.asUInt(16)` (instead of `16.W`) compile without error and produce undesired results.
        */
      implicit class fromBigIntToLiteral(bigint: BigInt) {
        /** Int to Bool conversion, allowing compact syntax like 1.B and 0.B
          */
        def B: Bool = bigint match { // scalastyle:ignore method.name
          case bigint if bigint == 0 => Bool.Lit(false)
          case bigint if bigint == 1 => Bool.Lit(true)
          case bigint => Builder.error(s"Cannot convert $bigint to Bool, must be 0 or 1"); Bool.Lit(false)
        }
        /** Int to UInt conversion, recommended style for constants.
          */
        def U: UInt = UInt.Lit(bigint, Width())  // scalastyle:ignore method.name
        /** Int to SInt conversion, recommended style for constants.
          */
        def S: SInt = SInt.Lit(bigint, Width())  // scalastyle:ignore method.name
        /** Int to UInt conversion with specified width, recommended style for constants.
          */
        def U(width: Width): UInt = UInt.Lit(bigint, width)  // scalastyle:ignore method.name
        /** Int to SInt conversion with specified width, recommended style for constants.
          */
        def S(width: Width): SInt = SInt.Lit(bigint, width)  // scalastyle:ignore method.name

        /** Int to UInt conversion, recommended style for variables.
          */
        def asUInt(): UInt = UInt.Lit(bigint, Width())
        /** Int to SInt conversion, recommended style for variables.
          */
        def asSInt(): SInt = SInt.Lit(bigint, Width())
        /** Int to UInt conversion with specified width, recommended style for variables.
          */
        def asUInt(width: Width): UInt = UInt.Lit(bigint, width)
        /** Int to SInt conversion with specified width, recommended style for variables.
          */
        def asSInt(width: Width): SInt = SInt.Lit(bigint, width)
      }

      implicit class fromIntToLiteral(int: Int) extends fromBigIntToLiteral(int)
      implicit class fromLongToLiteral(long: Long) extends fromBigIntToLiteral(long)

      implicit class fromStringToLiteral(str: String) {
        /** String to UInt parse, recommended style for constants.
          */
        def U: UInt = str.asUInt() // scalastyle:ignore method.name
        /** String to UInt parse with specified width, recommended style for constants.
          */
        def U(width: Width): UInt = str.asUInt(width) // scalastyle:ignore method.name

        /** String to UInt parse, recommended style for variables.
          */
        def asUInt(): UInt = {
          val bigInt = parse(str)
          UInt.Lit(bigInt, Width(bigInt.bitLength max 1))
        }
        /** String to UInt parse with specified width, recommended style for variables.
          */
        def asUInt(width: Width): UInt = UInt.Lit(parse(str), width)

        protected def parse(n: String) = {
          val (base, num) = n.splitAt(1)
          val radix = base match {
            case "x" | "h" => 16
            case "d" => 10
            case "o" => 8
            case "b" => 2
            case _ => Builder.error(s"Invalid base $base"); 2
          }
          BigInt(num.filterNot(_ == '_'), radix)
        }
      }

      implicit class fromBooleanToLiteral(boolean: Boolean) {
        /** Boolean to Bool conversion, recommended style for constants.
          */
        def B: Bool = Bool.Lit(boolean)  // scalastyle:ignore method.name

        /** Boolean to Bool conversion, recommended style for variables.
          */
        def asBool(): Bool = Bool.Lit(boolean)
      }

      // Fixed Point is experimental for now, but we alias the implicit conversion classes here
      //  to minimize disruption with existing code.
      implicit class fromDoubleToLiteral(double: Double) extends experimental.FixedPoint.Implicits.fromDoubleToLiteral(double)
      implicit class fromIntToBinaryPoint(int: Int) extends experimental.FixedPoint.Implicits.fromIntToBinaryPoint(int)

      implicit class fromIntToWidth(int: Int) {
        def W: Width = Width(int)  // scalastyle:ignore method.name
      }

  val WireInit = WireDefault

  implicit class AddDirectionToData[T<:Data](target: T) {
    @chiselRuntimeDeprecated
    @deprecated("Input(Data) should be used over Data.asInput", "chisel3")
    def asInput(implicit compileOptions: CompileOptions): T = Input(target)

    @chiselRuntimeDeprecated
    @deprecated("Output(Data) should be used over Data.asOutput", "chisel3")
    def asOutput(implicit compileOptions: CompileOptions): T = Output(target)

    @chiselRuntimeDeprecated
    @deprecated("Flipped(Data) should be used over Data.flip", "chisel3")
    def flip()(implicit compileOptions: CompileOptions): T = Flipped(target)
  }

  implicit class fromBitsable[T <: Data](data: T) {

    @chiselRuntimeDeprecated
    @deprecated("fromBits is deprecated, use asTypeOf instead", "chisel3")
    def fromBits(that: Bits)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      that.asTypeOf(data)
    }
  }

  implicit class cloneTypeable[T <: Data](target: T) {
    @chiselRuntimeDeprecated
    @deprecated("chiselCloneType is deprecated, use chiselTypeOf(...) to get the Chisel Type of a hardware object", "chisel3") // scalastyle:ignore line.size.limit
    def chiselCloneType: T = {
      target.cloneTypeFull.asInstanceOf[T]
    }
  }

  object Vec extends VecFactory {
    import scala.language.experimental.macros

    @chiselRuntimeDeprecated
    @deprecated("Vec argument order should be size, t; this will be removed by the official release", "chisel3")
    def apply[T <: Data](gen: T, n: Int)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.Vec[T] =
      apply(n, gen)

    @chiselRuntimeDeprecated
    @deprecated("Vec.fill(n)(gen) is deprecated, use VecInit(Seq.fill(n)(gen)) instead", "chisel3")
    def fill[T <: Data](n: Int)(gen: => T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.Vec[T] =
      apply(Seq.fill(n)(gen))

    def apply[T <: Data](elts: Seq[T]): chisel3.Vec[T] = macro VecTransform.apply_elts
    @chiselRuntimeDeprecated
    @deprecated("Vec(elts) is deprecated, use VecInit(elts) instead", "chisel3")
    def do_apply[T <: Data](elts: Seq[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.Vec[T] =
      chisel3.VecInit(elts)

    def apply[T <: Data](elt0: T, elts: T*): chisel3.Vec[T] = macro VecTransform.apply_elt0
    @chiselRuntimeDeprecated
    @deprecated("Vec(elt0, ...) is deprecated, use VecInit(elt0, ...) instead", "chisel3")
    def do_apply[T <: Data](elt0: T, elts: T*)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.Vec[T] =
      VecInit(elt0 +: elts.toSeq)

    def tabulate[T <: Data](n: Int)(gen: (Int) => T): chisel3.Vec[T] = macro VecTransform.tabulate
    @chiselRuntimeDeprecated
    @deprecated("Vec.tabulate(n)(gen) is deprecated, use VecInit.tabulate(n)(gen) instead", "chisel3")
    def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): chisel3.Vec[T] =
      chisel3.VecInit.tabulate(n)(gen)
  }

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
  trait UIntFactory extends UIntFactoryBase {
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
  trait SIntFactory extends SIntFactoryBase {
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
  trait BoolFactory extends BoolFactoryBase {
    /** Creates Bool literal.
     */
    @chiselRuntimeDeprecated
    @deprecated("use x.B", "chisel3, will be removed by end of 2017")
    def apply(x: Boolean): Bool = x.B
  }

  object Bits extends UIntFactory
  object UInt extends UIntFactory
  object SInt extends SIntFactory
  object Bool extends BoolFactory

  type InstanceId = internal.InstanceId

  @deprecated("Use 'SyncReadMem'", "chisel3")
  val SeqMem = chisel3.SyncReadMem
  @deprecated("Use 'SyncReadMem'", "chisel3")
  type SeqMem[T <: Data] = SyncReadMem[T]

  type Module = chisel3.experimental.LegacyModule

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

  type ChiselException = internal.ChiselException

  // Debugger/Tester access to internal Chisel data structures and methods.
  def getDataElements(a: Aggregate): Seq[Element] = {
    a.allElements
  }
  def getModulePorts(m: Module): Seq[Port] = m.getPorts
  // Invalidate API - a DontCare element for explicit assignment to outputs,
  //  indicating the signal is intentionally not driven.
  val DontCare = chisel3.internal.InternalDontCare

  class BindingException(message: String) extends ChiselException(message)
  /** A function expected a Chisel type but got a hardware object
    */
  case class ExpectedChiselTypeException(message: String) extends BindingException(message)
  /**A function expected a hardware object but got a Chisel type
    */
  case class ExpectedHardwareException(message: String) extends BindingException(message)
  /** An aggregate had a mix of specified and unspecified directionality children
    */
  case class MixedDirectionAggregateException(message: String) extends BindingException(message)
  /** Attempted to re-bind an already bound (directionality or hardware) object
    */
  case class RebindingException(message: String) extends BindingException(message)
  // Connection exceptions.
  case class BiConnectException(message: String) extends ChiselException(message)
  case class MonoConnectException(message: String) extends ChiselException(message)
}
