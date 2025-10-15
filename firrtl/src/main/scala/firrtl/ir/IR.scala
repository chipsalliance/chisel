// SPDX-License-Identifier: Apache-2.0

package firrtl
package ir

import firrtl.annotations.Annotation

import org.apache.commons.text.translate.{AggregateTranslator, JavaUnicodeEscaper, LookupTranslator}

import scala.collection.JavaConverters._

/** Intermediate Representation */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class FirrtlNode {
  def serialize: String
}

/** Use the [[firrtl.ir.Serializer]] to serialize this node. */
private[firrtl] trait UseSerializer extends FirrtlNode {
  def serialize: String = Serializer.serialize(this)
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Info extends FirrtlNode with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object NoInfo extends Info {
  override def toString: String = ""
}

/** Stores the string of a file info annotation in its escaped form. */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class FileInfo(escaped: String) extends Info {
  override def toString: String = " @[" + escaped + "]"
  def unescaped:         String = FileInfo.unescape(escaped)
  def split:             (String, String, String) = FileInfo.split(escaped)
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait HasName {
  val name: String
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait HasInfo {
  val info: Info
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
trait IsDeclaration extends HasName with HasInfo

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class PrimOp extends FirrtlNode {
  def serialize: String = this.toString
  def apply(args: Any*): DoPrim = {
    val groups = args.groupBy {
      case x: Expression => "exp"
      case x: BigInt     => "int"
      case x: Int        => "int"
      case other => "other"
    }
    val exprs = groups.getOrElse("exp", Nil).collect { case e: Expression =>
      e
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

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Expression extends FirrtlNode {
  def tpe: Type
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Reference(name: String, tpe: Type = UnknownType) extends Expression with HasName with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SubField(expr: Expression, name: String, tpe: Type = UnknownType)
    extends Expression
    with HasName
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SubIndex(expr: Expression, value: Int, tpe: Type) extends Expression with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SubAccess(expr: Expression, index: Expression, tpe: Type) extends Expression with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Mux(cond: Expression, tval: Expression, fval: Expression, tpe: Type = UnknownType)
    extends Expression
    with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ValidIf(cond: Expression, value: Expression, tpe: Type) extends Expression with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Literal extends Expression {
  val value: BigInt
  val width: Width
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class UIntLiteral(value: BigInt, width: Width) extends Literal with UseSerializer {
  def tpe = UIntType(width)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SIntLiteral(value: BigInt, width: Width) extends Literal with UseSerializer {
  def tpe = SIntType(width)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object SIntLiteral {
  def minWidth(value: BigInt): Width = IntWidth(value.bitLength + 1)
  def apply(value:    BigInt): SIntLiteral = new SIntLiteral(value, minWidth(value))
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntegerPropertyLiteral(value: BigInt) extends Literal with UseSerializer {
  def tpe = IntegerPropertyType
  val width = UnknownWidth
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DoublePropertyLiteral(value: Double) extends Expression with UseSerializer {
  def tpe = DoublePropertyType
  val width = UnknownWidth
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class StringPropertyLiteral(value: StringLit) extends Expression with UseSerializer {
  def tpe = StringPropertyType
  val width = UnknownWidth
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class BooleanPropertyLiteral(value: Boolean) extends Expression with UseSerializer {
  val tpe = BooleanPropertyType
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class PathPropertyLiteral(value: String) extends Expression with UseSerializer {
  val tpe = PathPropertyType
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SequencePropertyValue(tpe: Type, values: Seq[Expression]) extends Expression with UseSerializer

/** Property primitive operations.
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class PropPrimOp(name: String) {
  override def toString: String = name
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object IntegerAddOp extends PropPrimOp("integer_add")
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object IntegerMulOp extends PropPrimOp("integer_mul")
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object IntegerShrOp extends PropPrimOp("integer_shr")
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object IntegerShlOp extends PropPrimOp("integer_shl")
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object ListConcatOp extends PropPrimOp("list_concat")

/** Property expressions.
  *
  * Unlike other primitives, Property expressions serialize as a tree directly in their rvalue context.
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class PropExpr(info: Info, tpe: Type, op: PropPrimOp, args: Seq[Expression])
    extends Expression
    with UseSerializer {
  override def serialize: String = {
    val serializedOp = op.toString()
    val serializedArgs = args.map(_.serialize).mkString("(", ", ", ")")
    serializedOp + serializedArgs
  }
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DoPrim(op: PrimOp, args: Seq[Expression], consts: Seq[BigInt], tpe: Type)
    extends Expression
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Statement extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefWire(info: Info, name: String, tpe: Type) extends Statement with IsDeclaration with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefRegister(info: Info, name: String, tpe: Type, clock: Expression)
    extends Statement
    with IsDeclaration
    with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefRegisterWithReset(
  info:  Info,
  name:  String,
  tpe:   Type,
  clock: Expression,
  reset: Expression,
  init:  Expression
) extends Statement
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object DefInstance {
  def apply(name: String, module: String): DefInstance = DefInstance(NoInfo, name, module)
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefInstance(info: Info, name: String, module: String, tpe: Type = UnknownType)
    extends Statement
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefInstanceChoice(info: Info, name: String, default: String, option: String, choices: Seq[(String, String)])
    extends Statement
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefObject(info: Info, name: String, cls: String) extends Statement with IsDeclaration with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object ReadUnderWrite extends Enumeration {
  val Undefined = Value("undefined")
  val Old = Value("old")
  val New = Value("new")
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
  readUnderWrite: ReadUnderWrite.Value = ReadUnderWrite.Undefined
) extends Statement
    with IsDeclaration
    with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefNode(info: Info, name: String, value: Expression) extends Statement with IsDeclaration with UseSerializer

/** Record/bundle type definition that names a FIRRTL type with an alias name */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefTypeAlias(info: Info, name: String, tpe: Type) extends Statement with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Conditionally(info: Info, pred: Expression, conseq: Statement, alt: Statement)
    extends Statement
    with HasInfo
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object Block {
  def apply(head: Statement, tail: Statement*): Block = Block(head +: tail)
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Block(stmts: Seq[Statement]) extends Statement with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Connect(info: Info, loc: Expression, expr: Expression) extends Statement with HasInfo with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class PropAssign(info: Info, loc: Expression, expr: Expression) extends Statement with HasInfo with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IsInvalid(info: Info, expr: Expression) extends Statement with HasInfo with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Attach(info: Info, exprs: Seq[Expression]) extends Statement with HasInfo with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Stop(val info: Info, val ret: Int, val clk: Expression, val en: Expression, val name: String = "")
    extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Print(
  val info:   Info,
  val string: StringLit,
  val args:   Seq[Expression],
  val clk:    Expression,
  val en:     Expression,
  val name:   String = ""
) extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Fprint(
  val info:         Info,
  val filename:     StringLit,
  val filenameArgs: Seq[Expression],
  val string:       StringLit,
  val args:         Seq[Expression],
  val clk:          Expression,
  val en:           Expression,
  val name:         String = ""
) extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Flush(val info: Info, val filename: Option[StringLit], args: Seq[Expression], val clk: Expression)
    extends Statement
    with HasInfo
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeDefine(info: Info, sink: Expression, probeExpr: Expression) extends Statement with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeExpr(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class RWProbeExpr(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeRead(expr: Expression, tpe: Type = UnknownType) extends Expression with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeForceInitial(info: Info, probe: Expression, value: Expression) extends Statement with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeReleaseInitial(info: Info, probe: Expression) extends Statement with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeForce(info: Info, clock: Expression, cond: Expression, probe: Expression, value: Expression)
    extends Statement
    with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ProbeRelease(info: Info, clock: Expression, cond: Expression, probe: Expression)
    extends Statement
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class LayerConfig
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object LayerConfig {
  final case class Extract(outputDir: Option[String]) extends LayerConfig
  final case object Inline extends LayerConfig
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class Layer(info: Info, name: String, config: LayerConfig, body: Seq[Layer])
    extends FirrtlNode
    with IsDeclaration
    with UseSerializer {
  def outputDir: Option[String] = config match {
    case LayerConfig.Extract(outputDir) => outputDir
    case _                              => None
  }
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class LayerBlock(info: Info, layer: String, body: Statement) extends Statement with UseSerializer

// option and case
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefOption(info: Info, name: String, cases: Seq[DefOptionCase])
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefOptionCase(info: Info, name: String)

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Domain(info: Info, name: String)

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntrinsicExpr(intrinsic: String, args: Seq[Expression], params: Seq[Param], tpe: Type)
    extends Expression
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntrinsicStmt(
  info:      Info,
  intrinsic: String,
  args:      Seq[Expression],
  params:    Seq[Param],
  tpe:       Option[Type] = None
) extends Statement
    with UseSerializer

// formal
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object Formal extends Enumeration {
  val Assert = Value("assert")
  val Assume = Value("assume")
  val Cover = Value("cover")
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Verification(
  op:   Formal.Value,
  info: Info,
  clk:  Expression,
  pred: Expression,
  en:   Expression,
  msg:  StringLit,
  args: Seq[Expression],
  name: String = ""
) extends Statement
    with HasInfo
    with IsDeclaration
    with UseSerializer {
  require(op != Formal.Cover || args.isEmpty, "cover message cannot be used as a format string")
}
// end formal

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object EmptyStmt extends Statement with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Comment(text: String) extends Statement with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Width extends FirrtlNode {
  def +(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width + b.width)
    case _                          => UnknownWidth
  }
  def -(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width - b.width)
    case _                          => UnknownWidth
  }
  def max(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width.max(b.width))
    case _                          => UnknownWidth
  }
  def min(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width.min(b.width))
    case _                          => UnknownWidth
  }
}

/** Positive Integer Bit Width of a [[GroundType]] */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
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
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object UnknownWidth extends Width with UseSerializer

/** Orientation of [[Field]] */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Orientation extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object Default extends Orientation {
  def serialize: String = ""
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object Flip extends Orientation {
  def serialize: String = "flip "
}

/** Field of [[BundleType]] */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Field(name: String, flip: Orientation, tpe: Type) extends FirrtlNode with HasName with UseSerializer

/** Types of [[FirrtlNode]] */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class Type extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class GroundType extends Type {
  val width: Width
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
object GroundType {
  def unapply(ground: GroundType): Option[Width] = Some(ground.width)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class AggregateType extends Type

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class ProbeType(underlying: Type, color: Option[String] = None) extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
final case class RWProbeType(underlying: Type, color: Option[String] = None) extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ConstType(underlying: Type) extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class UIntType(width: Width) extends GroundType with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SIntType(width: Width) extends GroundType with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class BundleType(fields: Seq[Field]) extends AggregateType with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class VectorType(tpe: Type, size: Int) extends AggregateType with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object ClockType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
/* Abstract reset, will be inferred to UInt<1> or AsyncReset */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object ResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object AsyncResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class AnalogType(width: Width) extends GroundType with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class AliasType(name: String) extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class PropertyType extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object IntegerPropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object DoublePropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object StringPropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object BooleanPropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object PathPropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class SequencePropertyType(tpe: PropertyType) extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ClassPropertyType(name: String) extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object AnyRefPropertyType extends PropertyType

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DomainType(domain: String) extends Type with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DomainDefine(info: Info, sink: Expression, source: Expression) extends Statement with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object UnknownType extends Type with UseSerializer

/** [[Port]] Direction */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class Direction extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object Input extends Direction {
  def serialize: String = "input"
}
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case object Output extends Direction {
  def serialize: String = "output"
}

/** [[DefModule]] Port */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Port(info: Info, name: String, direction: Direction, tpe: Type, associations: Seq[String])
    extends FirrtlNode
    with IsDeclaration
    with UseSerializer

/** Parameters for external modules */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class Param extends FirrtlNode {
  def name: String
}

/** Integer (of any width) Parameter */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntParam(name: String, value: BigInt) extends Param with UseSerializer

/** IEEE Double Precision Parameter (for Verilog real) */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DoubleParam(name: String, value: Double) extends Param with UseSerializer

/** String Parameter */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class StringParam(name: String, value: StringLit) extends Param with UseSerializer

/** Raw String Parameter
  * Useful for Verilog type parameters
  * @note Firrtl doesn't guarantee anything about this String being legal in any backend
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class RawStringParam(name: String, value: String) extends Param with UseSerializer

/** Base class for modules */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
abstract class DefModule extends FirrtlNode with IsDeclaration {
  val info:  Info
  val name:  String
  val ports: Seq[Port]
}

/** Internal Module
  *
  * An instantiable hardware block
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Module(info: Info, name: String, public: Boolean, layers: Seq[String], ports: Seq[Port], body: Statement)
    extends DefModule
    with UseSerializer

/** External Module
  *
  * Generally used for Verilog black boxes
  * @param defname Defined name of the external module (ie. the name Firrtl will emit)
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ExtModule(
  info:    Info,
  name:    String,
  ports:   Seq[Port],
  defname: String,
  params:  Seq[Param],
  layers:  Seq[String]
) extends DefModule
    with UseSerializer

/** Intrinsic Module
  *
  * Used for compiler intrinsics.
  * @param intrinsic Defined intrinsic of the module
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntModule(info: Info, name: String, ports: Seq[Port], intrinsic: String, params: Seq[Param])
    extends DefModule
    with UseSerializer

/** Class definition
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DefClass(info: Info, name: String, ports: Seq[Port], body: Statement) extends DefModule with UseSerializer

/** Parameters for test declarations.
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
sealed abstract class TestParam extends FirrtlNode
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class IntTestParam(value: BigInt) extends TestParam with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DoubleTestParam(value: Double) extends TestParam with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class StringTestParam(value: String) extends TestParam with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class ArrayTestParam(value: Seq[TestParam]) extends TestParam with UseSerializer
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class MapTestParam(value: Map[String, TestParam]) extends TestParam with UseSerializer

/** Formal Test
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class FormalTest(info: Info, name: String, moduleName: String, params: MapTestParam)
    extends DefModule
    with UseSerializer {
  val ports: Seq[Port] = Seq.empty
}

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class Circuit(
  info:        Info,
  modules:     Seq[DefModule],
  main:        String,
  typeAliases: Seq[DefTypeAlias] = Seq.empty,
  layers:      Seq[Layer] = Seq.empty,
  options:     Seq[DefOption] = Seq.empty,
  domains:     Seq[Domain] = Seq.empty
) extends FirrtlNode
    with HasInfo
    with UseSerializer

@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class CircuitWithAnnos(circuit: Circuit, annotations: Seq[Annotation]) extends FirrtlNode with UseSerializer
