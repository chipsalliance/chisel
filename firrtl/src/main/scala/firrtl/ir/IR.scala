// SPDX-License-Identifier: Apache-2.0

package firrtl
package ir

import firrtl.annotations.Annotation

import dataclass.{data, since}
import org.apache.commons.text.translate.{AggregateTranslator, JavaUnicodeEscaper, LookupTranslator}

import scala.collection.JavaConverters._

/** Intermediate Representation */
abstract class FirrtlNode {
  def serialize: String
}

/** Use the [[firrtl.ir.Serializer]] to serialize this node. */
private[firrtl] trait UseSerializer extends FirrtlNode {
  def serialize: String = Serializer.serialize(this)
}

abstract class Info extends FirrtlNode with UseSerializer
case object NoInfo extends Info {
  override def toString: String = ""
}

/** Stores the string of a file info annotation in its escaped form. */
case class FileInfo(escaped: String) extends Info {
  override def toString: String = " @[" + escaped + "]"
  def unescaped:         String = FileInfo.unescape(escaped)
  def split:             (String, String, String) = FileInfo.split(escaped)
}

object FileInfo {
  def fromEscaped(s:   String): FileInfo = new FileInfo(s)
  def fromUnescaped(s: String): FileInfo = new FileInfo(escape(s))

  /** prepends a `\` to: `\`, `\n`, `\t` and `]` */
  def escape(s: String): String = {
    // Only run translator if String contains a character needing escaping,
    // Speeds up common case
    if (s.exists(char => escapePairs.contains(char))) {
      EscapeFirrtl.translate(s)
    } else {
      s
    }
  }

  /** removes the `\` in front of `\`, `\n`, `\t` and `]` */
  def unescape(s: String): String = {
    // Only run translator if String contains '\' which implies something needs unescaping
    // Speeds up common case
    if (s.contains('\\')) {
      UnescapeFirrtl.translate(s)
    } else {
      s
    }
  }

  /** take an already escaped String and do the additional escaping needed for Verilog comment */
  def escapedToVerilog(s: String) = EscapedToVerilog.translate(s)

  // custom `CharSequenceTranslator` for FIRRTL Info String escaping
  type CharMap = (CharSequence, CharSequence)
  private val escapePairs: Map[Char, String] = Map(
    '\\' -> "\\\\",
    '\n' -> "\\n",
    '\t' -> "\\t",
    ']' -> "\\]"
  )
  // Helper for constructing the LookupTranslators
  private def escapePairsCharSeq: Map[CharSequence, CharSequence] = escapePairs.map { case (k, v) => k.toString -> v }

  private val EscapeFirrtl = new LookupTranslator(escapePairsCharSeq.asJava)
  private val UnescapeFirrtl = new LookupTranslator(escapePairsCharSeq.map(_.swap).asJava)

  // EscapeFirrtl + EscapedToVerilog essentially does the same thing as running StringEscapeUtils.unescapeJava
  private val EscapedToVerilog = new AggregateTranslator(
    new LookupTranslator(
      Seq[CharMap](
        // ] is the one character that firrtl needs to be escaped that does not need to be escaped in
        "\\]" -> "]",
        "\"" -> "\\\"",
        // \n and \t are already escaped
        "\b" -> "\\b",
        "\f" -> "\\f",
        "\r" -> "\\r"
      ).toMap.asJava
    ),
    JavaUnicodeEscaper.outsideOf(32, 0x7f)
  )

  // Splits the FileInfo into its corresponding file, line, and column strings
  private def split(s: String): (String, String, String) = {
    s match {
      // Yield the three
      case FileInfoRegex(file, line, col) => (file, line, col)
      // Otherwise, just return the string itself and null for the other values
      case _ => (s, null, null)
    }
  }

  private val FileInfoRegex = """(?:([^\s]+)(?: (\d+)\:(\d+)))""".r
}

trait HasName {
  val name: String
}
trait HasInfo {
  val info: Info
}
trait IsDeclaration extends HasName with HasInfo

case class StringLit(string: String) extends FirrtlNode {
  import org.apache.commons.text.StringEscapeUtils

  /** Returns an escaped and quoted String */
  def escape: String = {
    "\"" + serialize + "\""
  }
  def serialize: String = StringEscapeUtils.escapeJava(string)

  /** Format the string for Verilog */
  def verilogFormat: StringLit = {
    StringLit(string.replaceAll("%x", "%h"))
  }

  /** Returns an escaped and quoted String */
  def verilogEscape: String = {
    // normalize to turn things like ö into o
    import java.text.Normalizer
    val normalized = Normalizer.normalize(string, Normalizer.Form.NFD)
    val ascii = normalized.flatMap(StringLit.toASCII)
    ascii.mkString("\"", "", "\"")
  }
}
object StringLit {
  import org.apache.commons.text.StringEscapeUtils

  /** Maps characters to ASCII for Verilog emission */
  private def toASCII(char: Char): List[Char] = char match {
    case nonASCII if !nonASCII.isValidByte => List('?')
    case '"'                               => List('\\', '"')
    case '\\'                              => List('\\', '\\')
    case c if c >= ' ' && c <= '~'         => List(c)
    case '\n'                              => List('\\', 'n')
    case '\t'                              => List('\\', 't')
    case _                                 => List('?')
  }

  /** Create a StringLit from a raw parsed String */
  def unescape(raw: String): StringLit = {
    StringLit(StringEscapeUtils.unescapeJava(raw))
  }
}

/** Primitive Operation
  *
  * See [[PrimOps]]
  */
abstract class PrimOp extends FirrtlNode {
  def serialize: String = this.toString
  def apply(args: Any*): DoPrim = {
    val groups = args.groupBy {
      case x: Expression => "exp"
      case x: BigInt     => "int"
      case x: Int        => "int"
      case other => "other"
    }
    val exprs = groups.getOrElse("exp", Nil).collect {
      case e: Expression => e
    }
    val consts = groups.getOrElse("int", Nil).map {
      _ match {
        case i: BigInt => i
        case i: Int    => BigInt(i)
      }
    }
    groups.get("other") match {
      case None    =>
      case Some(x) => sys.error(s"Shouldn't be here: $x")
    }
    DoPrim(this, exprs, consts, UnknownType)
  }
}

abstract class Expression extends FirrtlNode {
  def tpe: Type
}

case class Reference(name: String, tpe: Type = UnknownType) extends Expression with HasName with UseSerializer

case class SubField(expr: Expression, name: String, tpe: Type = UnknownType)
    extends Expression
    with HasName
    with UseSerializer

case class SubIndex(expr: Expression, value: Int, tpe: Type) extends Expression with UseSerializer

case class SubAccess(expr: Expression, index: Expression, tpe: Type) extends Expression with UseSerializer

case class Mux(cond: Expression, tval: Expression, fval: Expression, tpe: Type = UnknownType)
    extends Expression
    with UseSerializer
case class ValidIf(cond: Expression, value: Expression, tpe: Type) extends Expression with UseSerializer
abstract class Literal extends Expression {
  val value: BigInt
  val width: Width
}
case class UIntLiteral(value: BigInt, width: Width) extends Literal with UseSerializer {
  def tpe = UIntType(width)
}
object UIntLiteral {
  def minWidth(value: BigInt): Width = IntWidth(math.max(value.bitLength, 1))
  def apply(value:    BigInt): UIntLiteral = new UIntLiteral(value, minWidth(value))

  /** Utility to construct UIntLiterals masked by the width
    *
    * This supports truncating negative values as well as values that are too wide for the width
    */
  def masked(value: BigInt, width: IntWidth): UIntLiteral = {
    val mask = (BigInt(1) << width.width.toInt) - 1
    UIntLiteral(value & mask, width)
  }
}
case class SIntLiteral(value: BigInt, width: Width) extends Literal with UseSerializer {
  def tpe = SIntType(width)
}
object SIntLiteral {
  def minWidth(value: BigInt): Width = IntWidth(value.bitLength + 1)
  def apply(value:    BigInt): SIntLiteral = new SIntLiteral(value, minWidth(value))
}

case class IntegerPropertyLiteral(value: BigInt) extends Literal with UseSerializer {
  def tpe = IntegerPropertyType
  val width = UnknownWidth
}

case class DoublePropertyLiteral(value: Double) extends Expression with UseSerializer {
  def tpe = DoublePropertyType
  val width = UnknownWidth
}

case class StringPropertyLiteral(value: String) extends Expression with UseSerializer {
  def tpe = StringPropertyType
  val width = UnknownWidth
}

case class BooleanPropertyLiteral(value: Boolean) extends Expression with UseSerializer {
  val tpe = BooleanPropertyType
}

case class PathPropertyLiteral(value: String) extends Expression with UseSerializer {
  val tpe = PathPropertyType
}

case class SequencePropertyValue(tpe: Type, values: Seq[Expression]) extends Expression with UseSerializer

case class DoPrim(op: PrimOp, args: Seq[Expression], consts: Seq[BigInt], tpe: Type)
    extends Expression
    with UseSerializer

abstract class Statement extends FirrtlNode
case class DefWire(info: Info, name: String, tpe: Type) extends Statement with IsDeclaration with UseSerializer
case class DefRegister(
  info:  Info,
  name:  String,
  tpe:   Type,
  clock: Expression)
    extends Statement
    with IsDeclaration
    with UseSerializer
case class DefRegisterWithReset(
  info:  Info,
  name:  String,
  tpe:   Type,
  clock: Expression,
  reset: Expression,
  init:  Expression)
    extends Statement
    with IsDeclaration
    with UseSerializer

object DefInstance {
  def apply(name: String, module: String): DefInstance = DefInstance(NoInfo, name, module)
}

case class DefInstance(info: Info, name: String, module: String, tpe: Type = UnknownType)
    extends Statement
    with IsDeclaration
    with UseSerializer

case class DefObject(info: Info, name: String, cls: String) extends Statement with IsDeclaration with UseSerializer

object ReadUnderWrite extends Enumeration {
  val Undefined = Value("undefined")
  val Old = Value("old")
  val New = Value("new")
}

case class DefMemory(
  info:         Info,
  name:         String,
  dataType:     Type,
  depth:        BigInt,
  writeLatency: Int,
  readLatency:  Int,
  readers:      Seq[String],
  writers:      Seq[String],
  readwriters:  Seq[String],
  // TODO: handle read-under-write
  readUnderWrite: ReadUnderWrite.Value = ReadUnderWrite.Undefined)
    extends Statement
    with IsDeclaration
    with UseSerializer
case class DefNode(info: Info, name: String, value: Expression) extends Statement with IsDeclaration with UseSerializer

/** Record/bundle type definition that names a FIRRTL type with an alias name */
case class DefTypeAlias(info: Info, name: String, tpe: Type) extends Statement with UseSerializer

case class Conditionally(
  info:   Info,
  pred:   Expression,
  conseq: Statement,
  alt:    Statement)
    extends Statement
    with HasInfo
    with UseSerializer

object Block {
  def apply(head: Statement, tail: Statement*): Block = Block(head +: tail)
}

case class Block(stmts: Seq[Statement]) extends Statement with UseSerializer
case class Connect(info: Info, loc: Expression, expr: Expression) extends Statement with HasInfo with UseSerializer
case class PropAssign(info: Info, loc: Expression, expr: Expression) extends Statement with HasInfo with UseSerializer
case class IsInvalid(info: Info, expr: Expression) extends Statement with HasInfo with UseSerializer
case class Attach(info: Info, exprs: Seq[Expression]) extends Statement with HasInfo with UseSerializer

@data class Stop(info: Info, ret: Int, clk: Expression, en: Expression, @since("FIRRTL 1.5") name: String = "")
    extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer {
  def copy(info: Info = info, ret: Int = ret, clk: Expression = clk, en: Expression = en): Stop = {
    Stop(info, ret, clk, en, name)
  }
}
object Stop {
  def unapply(s: Stop): Some[(Info, Int, Expression, Expression)] = {
    Some((s.info, s.ret, s.clk, s.en))
  }
}
@data class Print(
  info:   Info,
  string: StringLit,
  args:   Seq[Expression],
  clk:    Expression,
  en:     Expression,
  @since("FIRRTL 1.5")
  name: String = "")
    extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer {
  def copy(
    info:   Info = info,
    string: StringLit = string,
    args:   Seq[Expression] = args,
    clk:    Expression = clk,
    en:     Expression = en
  ): Print = {
    Print(info, string, args, clk, en, name)
  }
}
object Print {
  def unapply(s: Print): Some[(Info, StringLit, Seq[Expression], Expression, Expression)] = {
    Some((s.info, s.string, s.args, s.clk, s.en))
  }
}

case class ProbeDefine(info: Info, sink: Expression, probeExpr: Expression) extends Statement with UseSerializer
case class ProbeExpr(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer

case class RWProbeExpr(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer
case class ProbeRead(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer

case class ProbeForceInitial(info: Info, probe: Expression, value: Expression) extends Statement with UseSerializer
case class ProbeReleaseInitial(info: Info, probe: Expression) extends Statement with UseSerializer
case class ProbeForce(info: Info, clock: Expression, cond: Expression, probe: Expression, value: Expression)
    extends Statement
    with UseSerializer
case class ProbeRelease(info: Info, clock: Expression, cond: Expression, probe: Expression)
    extends Statement
    with UseSerializer

object GroupConvention {
  sealed trait Type
  case object Bind extends Type {
    override def toString: String = "bind"
  }
}

case class GroupDeclare(info: Info, name: String, convention: GroupConvention.Type, body: Seq[GroupDeclare])
    extends FirrtlNode
    with IsDeclaration
    with UseSerializer
case class GroupDefine(info: Info, declaration: String, body: Statement) extends Statement with UseSerializer

// formal
object Formal extends Enumeration {
  val Assert = Value("assert")
  val Assume = Value("assume")
  val Cover = Value("cover")
}

@data class Verification(
  op:   Formal.Value,
  info: Info,
  clk:  Expression,
  pred: Expression,
  en:   Expression,
  msg:  StringLit,
  @since("FIRRTL 1.5")
  name: String = "")
    extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer {
  def copy(
    op:   Formal.Value = op,
    info: Info = info,
    clk:  Expression = clk,
    pred: Expression = pred,
    en:   Expression = en,
    msg:  StringLit = msg
  ): Verification = {
    Verification(op, info, clk, pred, en, msg, name)
  }
}
object Verification {
  def unapply(s: Verification): Some[(Formal.Value, Info, Expression, Expression, Expression, StringLit)] = {
    Some((s.op, s.info, s.clk, s.pred, s.en, s.msg))
  }
}
// end formal

case object EmptyStmt extends Statement with UseSerializer

abstract class Width extends FirrtlNode {
  def +(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width + b.width)
    case _ => UnknownWidth
  }
  def -(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width - b.width)
    case _ => UnknownWidth
  }
  def max(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width.max(b.width))
    case _ => UnknownWidth
  }
  def min(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width.min(b.width))
    case _ => UnknownWidth
  }
}

/** Positive Integer Bit Width of a [[GroundType]] */
object IntWidth {
  private val maxCached = 1024
  private val cache = new Array[IntWidth](maxCached + 1)
  def apply(width: BigInt): IntWidth = {
    if (0 <= width && width <= maxCached) {
      val i = width.toInt
      var w = cache(i)
      if (w eq null) {
        w = new IntWidth(width)
        cache(i) = w
      }
      w
    } else new IntWidth(width)
  }
  // For pattern matching
  def unapply(w: IntWidth): Option[BigInt] = Some(w.width)
}
class IntWidth(val width: BigInt) extends Width with Product with UseSerializer {
  override def equals(that: Any) = that match {
    case w: IntWidth => width == w.width
    case _ => false
  }
  override def hashCode = width.toInt
  override def productPrefix = "IntWidth"
  override def toString = s"$productPrefix($width)"
  def copy(width:    BigInt = width) = IntWidth(width)
  def canEqual(that: Any) = that.isInstanceOf[Width]
  def productArity = 1
  def productElement(int: Int) = int match {
    case 0 => width
    case _ => throw new IndexOutOfBoundsException
  }
}
case object UnknownWidth extends Width with UseSerializer

/** Orientation of [[Field]] */
abstract class Orientation extends FirrtlNode
case object Default extends Orientation {
  def serialize: String = ""
}
case object Flip extends Orientation {
  def serialize: String = "flip "
}

/** Field of [[BundleType]] */
case class Field(name: String, flip: Orientation, tpe: Type) extends FirrtlNode with HasName with UseSerializer

/** Types of [[FirrtlNode]] */
abstract class Type extends FirrtlNode
abstract class GroundType extends Type {
  val width: Width
}
object GroundType {
  def unapply(ground: GroundType): Option[Width] = Some(ground.width)
}
abstract class AggregateType extends Type

case class ProbeType(underlying: Type) extends Type with UseSerializer
case class RWProbeType(underlying: Type) extends Type with UseSerializer

case class ConstType(underlying: Type) extends Type with UseSerializer

case class UIntType(width: Width) extends GroundType with UseSerializer
case class SIntType(width: Width) extends GroundType with UseSerializer

case class BundleType(fields: Seq[Field]) extends AggregateType with UseSerializer
case class VectorType(tpe: Type, size: Int) extends AggregateType with UseSerializer
case object ClockType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
/* Abstract reset, will be inferred to UInt<1> or AsyncReset */
case object ResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
case object AsyncResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
case class AnalogType(width: Width) extends GroundType with UseSerializer

case class AliasType(name: String) extends Type with UseSerializer

sealed abstract class PropertyType extends Type with UseSerializer

case object IntegerPropertyType extends PropertyType

case object DoublePropertyType extends PropertyType

case object StringPropertyType extends PropertyType

case object BooleanPropertyType extends PropertyType

case object PathPropertyType extends PropertyType

case class SequencePropertyType(tpe: PropertyType) extends PropertyType

case class ClassPropertyType(name: String) extends PropertyType

case object AnyRefPropertyType extends PropertyType

case object UnknownType extends Type with UseSerializer

/** [[Port]] Direction */
sealed abstract class Direction extends FirrtlNode
case object Input extends Direction {
  def serialize: String = "input"
}
case object Output extends Direction {
  def serialize: String = "output"
}

/** [[DefModule]] Port */
case class Port(
  info:      Info,
  name:      String,
  direction: Direction,
  tpe:       Type)
    extends FirrtlNode
    with IsDeclaration
    with UseSerializer

/** Parameters for external modules */
sealed abstract class Param extends FirrtlNode {
  def name: String
}

/** Integer (of any width) Parameter */
case class IntParam(name: String, value: BigInt) extends Param with UseSerializer

/** IEEE Double Precision Parameter (for Verilog real) */
case class DoubleParam(name: String, value: Double) extends Param with UseSerializer

/** String Parameter */
case class StringParam(name: String, value: StringLit) extends Param with UseSerializer

/** Raw String Parameter
  * Useful for Verilog type parameters
  * @note Firrtl doesn't guarantee anything about this String being legal in any backend
  */
case class RawStringParam(name: String, value: String) extends Param with UseSerializer

/** Base class for modules */
abstract class DefModule extends FirrtlNode with IsDeclaration {
  val info:  Info
  val name:  String
  val ports: Seq[Port]
}

/** Internal Module
  *
  * An instantiable hardware block
  */
case class Module(info: Info, name: String, ports: Seq[Port], body: Statement) extends DefModule with UseSerializer

/** External Module
  *
  * Generally used for Verilog black boxes
  * @param defname Defined name of the external module (ie. the name Firrtl will emit)
  */
case class ExtModule(
  info:    Info,
  name:    String,
  ports:   Seq[Port],
  defname: String,
  params:  Seq[Param])
    extends DefModule
    with UseSerializer

/** Intrinsic Module
  *
  * Used for compiler intrinsics.
  * @param intrinsic Defined intrinsic of the module
  */
case class IntModule(
  info:      Info,
  name:      String,
  ports:     Seq[Port],
  intrinsic: String,
  params:    Seq[Param])
    extends DefModule
    with UseSerializer

/** Class definition
  */
case class DefClass(info: Info, name: String, ports: Seq[Port], body: Statement) extends DefModule with UseSerializer

case class Circuit(
  info:        Info,
  modules:     Seq[DefModule],
  main:        String,
  typeAliases: Seq[DefTypeAlias] = Seq.empty,
  groups:      Seq[GroupDeclare] = Seq.empty)
    extends FirrtlNode
    with HasInfo
    with UseSerializer

case class CircuitWithAnnos(circuit: Circuit, annotations: Seq[Annotation]) extends FirrtlNode with UseSerializer
