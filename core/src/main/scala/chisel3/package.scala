// SPDX-License-Identifier: Apache-2.0

import chisel3.internal.firrtl.BinaryPoint

/** This package contains the main chisel3 API.
 */
package object chisel3 {
  import internal.firrtl.{Port, Width}
  import internal.Builder

  import scala.language.implicitConversions

  /**
    * These implicit classes allow one to convert [[scala.Int]] or [[scala.BigInt]] to
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
    def B: Bool = bigint match {
      case bigint if bigint == 0 => Bool.Lit(false)
      case bigint if bigint == 1 => Bool.Lit(true)
      case bigint => Builder.error(s"Cannot convert $bigint to Bool, must be 0 or 1"); Bool.Lit(false)
    }
    /** Int to UInt conversion, recommended style for constants.
      */
    def U: UInt = UInt.Lit(bigint, Width())
    /** Int to SInt conversion, recommended style for constants.
      */
    def S: SInt = SInt.Lit(bigint, Width())
    /** Int to UInt conversion with specified width, recommended style for constants.
      */
    def U(width: Width): UInt = UInt.Lit(bigint, width)
    /** Int to SInt conversion with specified width, recommended style for constants.
      */
    def S(width: Width): SInt = SInt.Lit(bigint, width)

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
    def U: UInt = str.asUInt()
    /** String to UInt parse with specified width, recommended style for constants.
      */
    def U(width: Width): UInt = str.asUInt(width)

    /** String to UInt parse, recommended style for variables.
      */
    def asUInt(): UInt = {
      val bigInt = parse(str)
      UInt.Lit(bigInt, Width(bigInt.bitLength max 1))
    }
    /** String to UInt parse with specified width, recommended style for variables.
      */
    def asUInt(width: Width): UInt = UInt.Lit(parse(str), width)

    protected def parse(n: String): BigInt = {
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

  implicit class fromIntToBinaryPoint(int: Int) {
    def BP: BinaryPoint = BinaryPoint(int)
  }

  implicit class fromBooleanToLiteral(boolean: Boolean) {
    /** Boolean to Bool conversion, recommended style for constants.
      */
    def B: Bool = Bool.Lit(boolean)

    /** Boolean to Bool conversion, recommended style for variables.
      */
    def asBool(): Bool = Bool.Lit(boolean)
  }

  // Fixed Point is experimental for now, but we alias the implicit conversion classes here
  // to minimize disruption with existing code.
  implicit class fromDoubleToLiteral(double: Double)
    extends experimental.FixedPoint.Implicits.fromDoubleToLiteral(double)

  implicit class fromBigDecimalToLiteral(bigDecimal: BigDecimal)
    extends experimental.FixedPoint.Implicits.fromBigDecimalToLiteral(bigDecimal)

  // Interval is experimental for now, but we alias the implicit conversion classes here
  //  to minimize disruption with existing code.
  implicit class fromIntToLiteralInterval(int: Int)
    extends experimental.Interval.Implicits.fromIntToLiteralInterval(int)

  implicit class fromLongToLiteralInterval(long: Long)
    extends experimental.Interval.Implicits.fromLongToLiteralInterval(long)

  implicit class fromBigIntToLiteralInterval(bigInt: BigInt)
    extends experimental.Interval.Implicits.fromBigIntToLiteralInterval(bigInt)

  implicit class fromDoubleToLiteralInterval(double: Double)
    extends experimental.Interval.Implicits.fromDoubleToLiteralInterval(double)

  implicit class fromBigDecimalToLiteralInterval(bigDecimal: BigDecimal)
    extends experimental.Interval.Implicits.fromBigDecimalToLiteralInterval(bigDecimal)

  implicit class fromIntToWidth(int: Int) {
    def W: Width = Width(int)
  }

  val WireInit = WireDefault

  object Vec extends VecFactory

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

  object Bits extends UIntFactory
  object UInt extends UIntFactory
  object SInt extends SIntFactory
  object Bool extends BoolFactory

  type InstanceId = internal.InstanceId

  @deprecated("MultiIOModule is now just Module", "Chisel 3.5")
  type MultiIOModule = chisel3.Module

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
