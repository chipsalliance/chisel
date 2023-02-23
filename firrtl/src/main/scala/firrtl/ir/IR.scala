// SPDX-License-Identifier: Apache-2.0

package firrtl
package ir

import Utils.{dec2string, trim}
import dataclass.{data, since}
import firrtl.constraint.{Constraint, IsKnown, IsVar}
import org.apache.commons.text.translate.{AggregateTranslator, JavaUnicodeEscaper, LookupTranslator}

import scala.collection.JavaConverters._
import scala.math.BigDecimal.RoundingMode._

/** Intermediate Representation */
abstract class FirrtlNode {
  def serialize: String
}

/** Use the [[firrtl.ir.Serializer]] to serialize this node. */
private[firrtl] trait UseSerializer extends FirrtlNode {
  def serialize: String = Serializer.serialize(this)
}

abstract class Info extends FirrtlNode with UseSerializer {
  def ++(that: Info): Info
}
case object NoInfo extends Info {
  override def toString: String = ""
  def ++(that: Info): Info = that
}

/** Stores the string of a file info annotation in its escaped form. */
case class FileInfo(escaped: String) extends Info {
  override def toString: String = " @[" + escaped + "]"
  def ++(that: Info): Info = if (that == NoInfo) this else MultiInfo(Seq(this, that))
  def unescaped: String = FileInfo.unescape(escaped)
  def split:     (String, String, String) = FileInfo.split(escaped)
  @deprecated("Use FileInfo.unescaped instead. FileInfo.info will be removed in FIRRTL 1.5.", "FIRRTL 1.4")
  def info: StringLit = StringLit(this.unescaped)
}

object FileInfo {
  @deprecated("Use FileInfo.fromUnEscaped instead. FileInfo.apply will be removed in FIRRTL 1.5.", "FIRRTL 1.4")
  def apply(info:      StringLit): FileInfo = new FileInfo(escape(info.string))
  def fromEscaped(s:   String):    FileInfo = new FileInfo(s)
  def fromUnescaped(s: String):    FileInfo = new FileInfo(escape(s))

  /** prepends a `\` to: `\`, `\n`, `\t` and `]` */
  def escape(s: String): String = EscapeFirrtl.translate(s)

  /** removes the `\` in front of `\`, `\n`, `\t` and `]` */
  def unescape(s: String): String = UnescapeFirrtl.translate(s)

  /** take an already escaped String and do the additional escaping needed for Verilog comment */
  def escapedToVerilog(s: String) = EscapedToVerilog.translate(s)

  // custom `CharSequenceTranslator` for FIRRTL Info String escaping
  type CharMap = (CharSequence, CharSequence)
  private val EscapeFirrtl = new LookupTranslator(
    Seq[CharMap](
      "\\" -> "\\\\",
      "\n" -> "\\n",
      "\t" -> "\\t",
      "]" -> "\\]"
    ).toMap.asJava
  )
  private val UnescapeFirrtl = new LookupTranslator(
    Seq[CharMap](
      "\\\\" -> "\\",
      "\\n" -> "\n",
      "\\t" -> "\t",
      "\\]" -> "]"
    ).toMap.asJava
  )
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

case class MultiInfo(infos: Seq[Info]) extends Info {
  private def collectStrings(info: Info): Seq[String] = info match {
    case f: FileInfo => Seq(f.escaped)
    case MultiInfo(seq) => seq.flatMap(collectStrings)
    case NoInfo         => Seq.empty
  }
  override def toString: String = {
    val parts = collectStrings(this)
    if (parts.nonEmpty) parts.mkString(" @[", " ", "]")
    else ""
  }
  def ++(that: Info): Info = if (that == NoInfo) this else MultiInfo(infos :+ that)
  def flatten: Seq[FileInfo] = MultiInfo.compressInfo(MultiInfo.flattenInfo(infos))
}
object MultiInfo {
  def apply(infos: Info*) = {
    val infosx = infos.filterNot(_ == NoInfo)
    infosx.size match {
      case 0 => NoInfo
      case 1 => infos.head
      case _ => new MultiInfo(infos)
    }
  }

  // Internal utility for unpacking implicit MultiInfo structure for muxes
  // TODO should this be made into an API?
  private[firrtl] def demux(info: Info): (Info, Info, Info) = info match {
    case MultiInfo(infos) if infos.lengthCompare(3) == 0 => (infos(0), infos(1), infos(2))
    case other                                           => (other, NoInfo, NoInfo) // if not exactly 3, we don't know what to do
  }

  private def flattenInfo(infos: Seq[Info]): Seq[FileInfo] = infos.flatMap {
    case NoInfo => Seq()
    case f: FileInfo => Seq(f)
    case MultiInfo(infos) => flattenInfo(infos)
  }

  private def compressInfo(infos: Seq[FileInfo]): Seq[FileInfo] = {
    // Sort infos by file name, then line number, then column number
    val sorted = infos.sortWith((A, B) => {
      val (fileA, lineA, colA) = A.split
      val (fileB, lineB, colB) = B.split

      // A FileInfo with no line nor column number should be sorted to the beginning of
      // the sequence of FileInfos that share its fileName. This way, they can be immediately
      // skipped by the algorithm
      if (lineA == null || lineB == null)
        fileA <= fileB && lineA == null
      else
        // Default comparison
        fileA <= fileB &&
        lineA <= lineB &&
        colA <= colB
    })

    // Holds the current file/line being parsed.
    var currentFile = ""
    var currentLine = ""

    // Holds any compressed line/column numbers to be appended after the file name
    var locators = new StringBuilder
    // Holds all encountered columns that exist within the current line
    var columns: Seq[String] = Seq()

    var out: Seq[FileInfo] = Seq()

    // Helper function to append the contents of the columns Seq to the locators buffer.
    def serializeColumns: Unit = {
      if (columns.size == 0)
        return

      var columnsList = columns.mkString(",")
      // Wrap the columns in curly braces if it contains more than one entry
      if (columns.size > 1)
        columnsList = '{' + columnsList + '}'

      // If there already exists line/column numbers in the buffer, delimit the new
      // info with a space
      if (locators.nonEmpty)
        locators ++= " "

      locators ++= s"$currentLine:$columnsList"
    }

    for (info <- sorted) {
      val (file, line, col) = info.split

      // Only process file infos that match the pattern *fully*, so that all 3 capture groups were
      // matched
      // TODO: Enforce a specific format for FileInfos (file.name line1:{col1,col2} line2:{col3,col4}...),
      // so that this code can run with the assumption that all FileInfos obey that format.
      if (line != null && col != null) {
        // If we encounter a new file, yield the current compressed info
        if (file != currentFile) {
          if (currentFile.nonEmpty) {
            serializeColumns
            out :+= FileInfo.fromEscaped(s"$currentFile $locators")
          }

          // Reset all tracking variables to now track the new file
          currentFile = file
          currentLine = ""
          locators.clear()
          columns = Seq()
        }

        // If we encounter a new line, append the current columns to the line buffer.
        if (line != currentLine) {
          if (currentLine.nonEmpty)
            serializeColumns
          // Track the new current line
          currentLine = line
          columns = Seq()
        }

        columns :+= col
      } else
        // Add in the uncompressed FileInfo
        out :+= info
    }

    // Serialize any remaining column info that was parsed by the loop
    serializeColumns

    // Append the remaining FileInfo if one was parsed
    if (currentFile.nonEmpty)
      out :+= FileInfo.fromEscaped(s"$currentFile $locators")

    out
  }
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

/** Represents reference-like expression nodes: SubField, SubIndex, SubAccess and Reference
  * The following fields can be cast to RefLikeExpression in every well formed firrtl AST:
  * - SubField.expr, SubIndex.expr, SubAccess.expr
  * - IsInvalid.expr, Connect.loc, PartialConnect.loc
  * - Attach.exprs
  */
sealed trait RefLikeExpression extends Expression { def flow: Flow }

/** Represents a statement that can be referenced in a firrtl expression.
  * This explicitly excludes named side-effecting statements like Print, Stop and Verification.
  * Note: This trait cannot be sealed since the memory ports are declared in WIR.scala.
  *       Once we fully remove all WIR, this trait could be sealed.
  */
trait CanBeReferenced

object Reference {

  /** Creates a Reference from a Wire */
  def apply(wire: DefWire): Reference = Reference(wire.name, wire.tpe, WireKind, UnknownFlow)

  /** Creates a Reference from a Register */
  def apply(reg: DefRegister): Reference = Reference(reg.name, reg.tpe, RegKind, UnknownFlow)

  /** Creates a Reference from a Node */
  def apply(node: DefNode): Reference = Reference(node.name, node.value.tpe, NodeKind, SourceFlow)

  /** Creates a Reference from a Port */
  def apply(port: Port): Reference = Reference(port.name, port.tpe, PortKind, UnknownFlow)

  /** Creates a Reference from a DefInstance */
  def apply(i: DefInstance): Reference = Reference(i.name, i.tpe, InstanceKind, UnknownFlow)

  /** Creates a Reference from a DefMemory */
  def apply(mem: DefMemory): Reference = Reference(mem.name, passes.MemPortUtils.memType(mem), MemKind, UnknownFlow)
}

case class Reference(name: String, tpe: Type = UnknownType, kind: Kind = UnknownKind, flow: Flow = UnknownFlow)
    extends Expression
    with HasName
    with UseSerializer
    with RefLikeExpression

case class SubField(expr: Expression, name: String, tpe: Type = UnknownType, flow: Flow = UnknownFlow)
    extends Expression
    with HasName
    with UseSerializer
    with RefLikeExpression

case class SubIndex(expr: Expression, value: Int, tpe: Type, flow: Flow = UnknownFlow)
    extends Expression
    with UseSerializer
    with RefLikeExpression

case class SubAccess(expr: Expression, index: Expression, tpe: Type, flow: Flow = UnknownFlow)
    extends Expression
    with UseSerializer
    with RefLikeExpression

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
case class DoPrim(op: PrimOp, args: Seq[Expression], consts: Seq[BigInt], tpe: Type)
    extends Expression
    with UseSerializer

abstract class Statement extends FirrtlNode
case class DefWire(info: Info, name: String, tpe: Type)
    extends Statement
    with IsDeclaration
    with CanBeReferenced
    with UseSerializer
case class DefRegister(
  info:  Info,
  name:  String,
  tpe:   Type,
  clock: Expression,
  reset: Expression,
  init:  Expression)
    extends Statement
    with IsDeclaration
    with CanBeReferenced
    with UseSerializer

object DefInstance {
  def apply(name: String, module: String): DefInstance = DefInstance(NoInfo, name, module)
}

case class DefInstance(info: Info, name: String, module: String, tpe: Type = UnknownType)
    extends Statement
    with IsDeclaration
    with CanBeReferenced
    with UseSerializer

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
    with CanBeReferenced
    with UseSerializer
case class DefNode(info: Info, name: String, value: Expression)
    extends Statement
    with IsDeclaration
    with CanBeReferenced
    with UseSerializer
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
case class PartialConnect(info: Info, loc: Expression, expr: Expression)
    extends Statement
    with HasInfo
    with UseSerializer
case class Connect(info: Info, loc: Expression, expr: Expression) extends Statement with HasInfo with UseSerializer
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

case class CalcWidth(arg: Constraint) extends Width with UseSerializer

case class VarWidth(name: String) extends Width with IsVar {
  override def serialize: String = name
}

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

/** Bounds of [[IntervalType]] */

trait Bound extends Constraint
case object UnknownBound extends Bound {
  def serialize: String = Serializer.serialize(this)
  def map(f: Constraint => Constraint): Constraint = this
  override def reduce(): Constraint = this
  val children = Vector()
}
case class CalcBound(arg: Constraint) extends Bound {
  def serialize: String = Serializer.serialize(this)
  def map(f: Constraint => Constraint): Constraint = f(arg)
  override def reduce(): Constraint = arg
  val children = Vector(arg)
}
case class VarBound(name: String) extends IsVar with Bound {
  override def serialize: String = Serializer.serialize(this)
}
object KnownBound {
  def unapply(b: Constraint): Option[BigDecimal] = b match {
    case k: IsKnown => Some(k.value)
    case _ => None
  }
  def unapply(b: Bound): Option[BigDecimal] = b match {
    case k: IsKnown => Some(k.value)
    case _ => None
  }
}
case class Open(value: BigDecimal) extends IsKnown with Bound {
  def serialize: String = Serializer.serialize(this)
  def +(that: IsKnown): IsKnown = Open(value + that.value)
  def *(that: IsKnown): IsKnown = that match {
    case Closed(x) if x == 0 => Closed(x)
    case _                   => Open(value * that.value)
  }
  def min(that: IsKnown): IsKnown = if (value < that.value) this else that
  def max(that: IsKnown): IsKnown = if (value > that.value) this else that
  def neg:   IsKnown = Open(-value)
  def floor: IsKnown = Open(value.setScale(0, BigDecimal.RoundingMode.FLOOR))
  def pow: IsKnown =
    if (value.isBinaryDouble) Open(BigDecimal(BigInt(1) << value.toInt)) else sys.error("Shouldn't be here")
}
case class Closed(value: BigDecimal) extends IsKnown with Bound {
  def serialize: String = Serializer.serialize(this)
  def +(that: IsKnown): IsKnown = that match {
    case Open(x)   => Open(value + x)
    case Closed(x) => Closed(value + x)
  }
  def *(that: IsKnown): IsKnown = that match {
    case IsKnown(x) if value == BigInt(0) => Closed(0)
    case Open(x)                          => Open(value * x)
    case Closed(x)                        => Closed(value * x)
  }
  def min(that: IsKnown): IsKnown = if (value <= that.value) this else that
  def max(that: IsKnown): IsKnown = if (value >= that.value) this else that
  def neg:   IsKnown = Closed(-value)
  def floor: IsKnown = Closed(value.setScale(0, BigDecimal.RoundingMode.FLOOR))
  def pow: IsKnown =
    if (value.isBinaryDouble) Closed(BigDecimal(BigInt(1) << value.toInt)) else sys.error("Shouldn't be here")
}

/** Types of [[FirrtlNode]] */
abstract class Type extends FirrtlNode
abstract class GroundType extends Type {
  val width: Width
}
object GroundType {
  def unapply(ground: GroundType): Option[Width] = Some(ground.width)
}
abstract class AggregateType extends Type
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
    with CanBeReferenced
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

case class Circuit(info: Info, modules: Seq[DefModule], main: String) extends FirrtlNode with HasInfo with UseSerializer
