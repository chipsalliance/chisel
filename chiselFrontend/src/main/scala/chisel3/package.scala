package object chisel3 {    // scalastyle:ignore package.object.name
  import core.CompileOptions
  import internal.firrtl.{BinaryPoint, Port, Width}
  import internal.sourceinfo.{SourceInfo, VecTransform}
  import internal.{Builder, chiselRuntimeDeprecated, throwException}

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

      //scalastyle:off method.name
      implicit class fromDoubleToLiteral(double: Double) {
        @deprecated("Use notation <double>.F(<binary_point>.BP) instead", "chisel3")
        def F(binaryPoint: Int): FixedPoint = FixedPoint.fromDouble(double, binaryPoint = binaryPoint)
        def F(binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, Width(), binaryPoint)
        }
        def F(width: Width, binaryPoint: BinaryPoint): FixedPoint = {
          FixedPoint.fromDouble(double, width, binaryPoint)
        }
      }

      implicit class fromIntToWidth(int: Int) {
        def W: Width = Width(int)  // scalastyle:ignore method.name
      }

      implicit class fromIntToBinaryPoint(int: Int) {
        def BP: BinaryPoint = BinaryPoint(int)  // scalastyle:ignore method.name
      }

      // These provide temporary compatibility for those who foolishly imported from chisel3.core
      @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
        " Use chisel3.experimental.RawModule instead.", "since the beginning of time")
      type UserModule = core.RawModule
      @deprecated("Avoid importing from chisel3.core, these are not public APIs and may change at any time. " +
        "Use chisel3.experimental.MultiIOModule instead.", "since the beginning of time")
      type ImplicitModule = core.MultiIOModule

  object Wire extends WireFactory {

    @chiselRuntimeDeprecated
    @deprecated("Wire(init=init) is deprecated, use WireDefault(init) instead", "chisel3")
    def apply[T <: Data](dummy: Int = 0, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      WireDefault(init)

    @chiselRuntimeDeprecated
    @deprecated("Wire(t, init) is deprecated, use WireDefault(t, init) instead", "chisel3")
    def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      WireDefault(t, init)

    @chiselRuntimeDeprecated
    @deprecated("Wire(t, init) is deprecated, use WireDefault(t, init) instead", "chisel3")
    def apply[T <: Data](t: T, init: DontCare.type)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      WireDefault(t, init)
  }
  val WireInit = WireDefault

  val Clock = core.Clock
  type Clock = core.Clock

  // Clock and reset scoping functions
  val withClockAndReset = core.withClockAndReset
  val withClock = core.withClock
  val withReset = core.withReset

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

  type Aggregate = core.Aggregate
  object Vec extends core.VecFactory {
    import scala.language.experimental.macros

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
      core.VecInit(elts)

    def apply[T <: Data](elt0: T, elts: T*): Vec[T] = macro VecTransform.apply_elt0
    @chiselRuntimeDeprecated
    @deprecated("Vec(elt0, ...) is deprecated, use VecInit(elt0, ...) instead", "chisel3")
    def do_apply[T <: Data](elt0: T, elts: T*)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      core.VecInit(elt0 +: elts.toSeq)

    def tabulate[T <: Data](n: Int)(gen: (Int) => T): Vec[T] = macro VecTransform.tabulate
    @chiselRuntimeDeprecated
    @deprecated("Vec.tabulate(n)(gen) is deprecated, use VecInit.tabulate(n)(gen) instead", "chisel3")
    def do_tabulate[T <: Data](n: Int)(gen: (Int) => T)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Vec[T] =
      core.VecInit.tabulate(n)(gen)
  }
  val VecInit = core.VecInit
  type Vec[T <: Data] = core.Vec[T]
  type VecLike[T <: Data] = core.VecLike[T]
  type Bundle = core.Bundle
  type IgnoreSeqInBundle = core.IgnoreSeqInBundle
  type Record = core.Record

  val assert = core.assert

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
  val Mux = core.Mux

  type BlackBox = core.BlackBox

  type InstanceId = internal.InstanceId

  val Mem = core.Mem
  type MemBase[T <: Data] = core.MemBase[T]
  type Mem[T <: Data] = core.Mem[T]
  val SyncReadMem = core.SyncReadMem
  type SyncReadMem[T <: Data] = core.SyncReadMem[T]

  @deprecated("Use 'SyncReadMem'", "chisel3")
  val SeqMem = core.SyncReadMem
  @deprecated("Use 'SyncReadMem'", "chisel3")
  type SeqMem[T <: Data] = core.SyncReadMem[T]

  val Module = core.Module
  type Module = core.LegacyModule

  val printf = core.printf

  val RegNext = core.RegNext
  val RegInit = core.RegInit
  object Reg {

    // Passthrough for core.Reg
    // TODO: make val Reg = core.Reg once we eliminate the legacy Reg constructor
    def apply[T <: Data](t: T)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T =
      core.Reg(t)

    @chiselRuntimeDeprecated
    @deprecated("Use Reg(t), RegNext(next, [init]) or RegInit([t], init) instead", "chisel3")
    def apply[T <: Data](t: T = null, next: T = null, init: T = null)
        (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): T = {
      if (t ne null) {
        val reg = if (init ne null) {
          RegInit(t, init)
        } else {
          core.Reg(t)
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

  val when = core.when
  type WhenContext = core.WhenContext

  type Printable = core.Printable
  val Printable = core.Printable
  type Printables = core.Printables
  val Printables = core.Printables
  type PString = core.PString
  val PString = core.PString
  type FirrtlFormat = core.FirrtlFormat
  val FirrtlFormat = core.FirrtlFormat
  type Decimal = core.Decimal
  val Decimal = core.Decimal
  type Hexadecimal = core.Hexadecimal
  val Hexadecimal = core.Hexadecimal
  type Binary = core.Binary
  val Binary = core.Binary
  type Character = core.Character
  val Character = core.Character
  type Name = core.Name
  val Name = core.Name
  type FullName = core.FullName
  val FullName = core.FullName
  val Percent = core.Percent

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

  implicit class fromUIntToBitPatComparable(x: UInt) extends SourceInfoDoc {
    import internal.sourceinfo.{SourceInfo, SourceInfoTransform}

    import scala.language.experimental.macros

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


  type ChiselException = internal.ChiselException

  // Debugger/Tester access to internal Chisel data structures and methods.
  def getDataElements(a: Aggregate): Seq[Element] = {
    a.allElements
  }
  def getModulePorts(m: Module): Seq[Port] = m.getPorts

  /** Package for experimental features, which may have their API changed, be removed, etc.
    *
    * Because its contents won't necessarily have the same level of stability and support as
    * non-experimental, you must explicitly import this package to use its contents.
    */
  object experimental {  // scalastyle:ignore object.name
    type Param = core.Param
    type IntParam = core.IntParam
    val IntParam = core.IntParam
    type DoubleParam = core.DoubleParam
    val DoubleParam = core.DoubleParam
    type StringParam = core.StringParam
    val StringParam = core.StringParam
    type RawParam = core.RawParam
    val RawParam = core.RawParam

    val attach = core.attach

    type ChiselEnum = core.EnumFactory
    val EnumAnnotations = core.EnumAnnotations

    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withClockAndReset = core.withClockAndReset
    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withClock = core.withClock
    @deprecated("Use the version in chisel3._", "chisel3.2")
    val withReset = core.withReset

    val dontTouch = core.dontTouch

    type BaseModule = core.BaseModule
    type RawModule = core.RawModule
    type MultiIOModule = core.MultiIOModule
    type ExtModule = core.ExtModule

    val IO = core.IO

    // Rocket Chip-style clonemodule

    /** A record containing the results of CloneModuleAsRecord
      * The apply method is retrieves the element with the supplied name.
      */
    type ClonePorts = core.BaseModule.ClonePorts

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
      def apply(proto: BaseModule)(implicit sourceInfo: internal.sourceinfo.SourceInfo, compileOptions: core.CompileOptions): ClonePorts = { // scalastyle:ignore line.size.limit
        core.BaseModule.cloneIORecord(proto)
      }
    }

    // Implicit conversions for BlackBox Parameters
    implicit def fromIntToIntParam(x: Int): IntParam = IntParam(BigInt(x))
    implicit def fromLongToIntParam(x: Long): IntParam = IntParam(BigInt(x))
    implicit def fromBigIntToIntParam(x: BigInt): IntParam = IntParam(x)
    implicit def fromDoubleToDoubleParam(x: Double): DoubleParam = DoubleParam(x)
    implicit def fromStringToStringParam(x: String): StringParam = StringParam(x)

    type ChiselAnnotation = core.ChiselAnnotation
    val ChiselAnnotation = core.ChiselAnnotation
    type RunFirrtlTransform = core.RunFirrtlTransform

    val annotate = core.annotate

    val requireIsHardware = core.requireIsHardware
    val requireIsChiselType = core.requireIsChiselType
    type Direction = ActualDirection
    val Direction = ActualDirection

    implicit class ChiselRange(val sc: StringContext) extends AnyVal {
      import internal.firrtl.NumericBound

      import scala.language.experimental.macros

      /** Specifies a range using mathematical range notation. Variables can be interpolated using
        * standard string interpolation syntax.
        * @example {{{
        * UInt(range"[0, 2)")
        * UInt(range"[0, \$myInt)")
        * UInt(range"[0, \${myInt + 2})")
        * }}}
        */
      def range(args: Any*): (NumericBound[Int], NumericBound[Int]) = macro internal.RangeTransform.apply
    }

    class dump extends internal.naming.dump  // scalastyle:ignore class.name
    class treedump extends internal.naming.treedump  // scalastyle:ignore class.name
    class chiselName extends internal.naming.chiselName  // scalastyle:ignore class.name
  }
}
