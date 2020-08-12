// See LICENSE for license details.

package firrtl
package ir

import Utils.{dec2string, trim}
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
  @deprecated("Use FileInfo.unescaped instead. FileInfo.info will be removed in FIRRTL 1.5.", "FIRRTL 1.4")
  def info: StringLit = StringLit(this.unescaped)
}

object FileInfo {
  @deprecated("Use FileInfo.fromUnEscaped instead. FileInfo.apply will be removed in FIRRTL 1.5.", "FIRRTL 1.4")
  def apply(info: StringLit): FileInfo = new FileInfo(escape(info.string))
  def fromEscaped(s: String): FileInfo = new FileInfo(s)
  def fromUnescaped(s: String): FileInfo = new FileInfo(escape(s))
  /** prepends a `\` to: `\`, `\n`, `\t` and `]` */
  def escape(s: String): String = EscapeFirrtl.translate(s)
  /** removes the `\` in front of `\`, `\n`, `\t` and `]` */
  def unescape(s: String): String = UnescapeFirrtl.translate(s)
  /** take an already escaped String and do the additional escaping needed for Verilog comment */
  def escapedToVerilog(s: String) = EscapedToVerilog.translate(s)

  // custom `CharSequenceTranslator` for FIRRTL Info String escaping
  type CharMap = (CharSequence, CharSequence)
  private val EscapeFirrtl = new LookupTranslator(Seq[CharMap](
    "\\" -> "\\\\",
    "\n" -> "\\n",
    "\t" -> "\\t",
    "]" -> "\\]"
  ).toMap.asJava)
  private val UnescapeFirrtl = new LookupTranslator(Seq[CharMap](
    "\\\\" -> "\\",
    "\\n" -> "\n",
    "\\t" -> "\t",
    "\\]" -> "]"
  ).toMap.asJava)
  // EscapeFirrtl + EscapedToVerilog essentially does the same thing as running StringEscapeUtils.unescapeJava
  private val EscapedToVerilog = new AggregateTranslator(
    new LookupTranslator(Seq[CharMap](
      // ] is the one character that firrtl needs to be escaped that does not need to be escaped in
      "\\]" -> "]",
      "\"" -> "\\\"",
      // \n and \t are already escaped
      "\b" -> "\\b",
      "\f" -> "\\f",
      "\r" -> "\\r"
    ).toMap.asJava),
    JavaUnicodeEscaper.outsideOf(32, 0x7f)
  )

}

case class MultiInfo(infos: Seq[Info]) extends Info {
  private def collectStrings(info: Info): Seq[String] = info match {
    case f : FileInfo => Seq(f.escaped)
    case MultiInfo(seq) => seq flatMap collectStrings
    case NoInfo => Seq.empty
  }
  override def toString: String = {
    val parts = collectStrings(this)
    if (parts.nonEmpty) parts.mkString(" @[", " ", "]")
    else ""
  }
  def ++(that: Info): Info = if (that == NoInfo) this else MultiInfo(infos :+ that)
  def flatten: Seq[FileInfo] = MultiInfo.flattenInfo(infos)
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
    case other => (other, NoInfo, NoInfo) // if not exactly 3, we don't know what to do
  }
  
  private def flattenInfo(infos: Seq[Info]): Seq[FileInfo] = infos.flatMap {
    case NoInfo => Seq()
    case f : FileInfo => Seq(f)
    case MultiInfo(infos) => flattenInfo(infos)
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
    val ascii = normalized flatMap StringLit.toASCII
    ascii.mkString("\"", "", "\"")
  }
}
object StringLit {
  import org.apache.commons.text.StringEscapeUtils
  /** Maps characters to ASCII for Verilog emission */
  private def toASCII(char: Char): List[Char] = char match {
    case nonASCII if !nonASCII.isValidByte => List('?')
    case '"' => List('\\', '"')
    case '\\' => List('\\', '\\')
    case c if c >= ' ' && c <= '~' => List(c)
    case '\n' => List('\\', 'n')
    case '\t' => List('\\', 't')
    case _ => List('?')
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
  def propagateType(e: DoPrim): Type = UnknownType
  def apply(args: Any*): DoPrim = {
    val groups = args.groupBy {
      case x: Expression => "exp"
      case x: BigInt => "int"
      case x: Int => "int"
      case other => "other"
    }
    val exprs = groups.getOrElse("exp", Nil).collect {
      case e: Expression => e
    }
    val consts = groups.getOrElse("int", Nil).map {
      _ match {
        case i: BigInt => i
        case i: Int => BigInt(i)
      }
    }
    groups.get("other") match {
      case None =>
      case Some(x) => sys.error(s"Shouldn't be here: $x")
    }
    DoPrim(this, exprs, consts, UnknownType)
  }
}

abstract class Expression extends FirrtlNode {
  def tpe: Type
  def mapExpr(f: Expression => Expression): Expression
  def mapType(f: Type => Type): Expression
  def mapWidth(f: Width => Width): Expression
  def foreachExpr(f: Expression => Unit): Unit
  def foreachType(f: Type => Unit): Unit
  def foreachWidth(f: Width => Unit): Unit
}

/** Represents reference-like expression nodes: SubField, SubIndex, SubAccess and Reference
  * The following fields can be cast to RefLikeExpression in every well formed firrtl AST:
  * - SubField.expr, SubIndex.expr, SubAccess.expr
  * - IsInvalid.expr, Connect.loc, PartialConnect.loc
  * - Attach.exprs
  */
sealed trait RefLikeExpression extends Expression { def flow: Flow }

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
    extends Expression with HasName with UseSerializer with RefLikeExpression {
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}

case class SubField(expr: Expression, name: String, tpe: Type = UnknownType, flow: Flow = UnknownFlow)
    extends Expression with HasName with UseSerializer with RefLikeExpression {
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = f(expr)
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}

case class SubIndex(expr: Expression, value: Int, tpe: Type, flow: Flow = UnknownFlow)
    extends Expression with UseSerializer with RefLikeExpression {
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = f(expr)
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}

case class SubAccess(expr: Expression, index: Expression, tpe: Type, flow: Flow = UnknownFlow)
    extends Expression with UseSerializer with RefLikeExpression {
  def mapExpr(f: Expression => Expression): Expression = this.copy(expr = f(expr), index = f(index))
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = { f(expr); f(index) }
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}

case class Mux(cond: Expression, tval: Expression, fval: Expression, tpe: Type = UnknownType)
  extends Expression with UseSerializer {
  def mapExpr(f: Expression => Expression): Expression = Mux(f(cond), f(tval), f(fval), tpe)
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = { f(cond); f(tval); f(fval) }
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}
case class ValidIf(cond: Expression, value: Expression, tpe: Type) extends Expression with UseSerializer {
  def mapExpr(f: Expression => Expression): Expression = ValidIf(f(cond), f(value), tpe)
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = { f(cond); f(value) }
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}
abstract class Literal extends Expression {
  val value: BigInt
  val width: Width
}
case class UIntLiteral(value: BigInt, width: Width) extends Literal with UseSerializer {
  def tpe = UIntType(width)
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = UIntLiteral(value, f(width))
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = ()
  def foreachWidth(f: Width => Unit): Unit = f(width)
}
object UIntLiteral {
  def minWidth(value: BigInt): Width = IntWidth(math.max(value.bitLength, 1))
  def apply(value: BigInt): UIntLiteral = new UIntLiteral(value, minWidth(value))

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
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = SIntLiteral(value, f(width))
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = ()
  def foreachWidth(f: Width => Unit): Unit = f(width)
}
object SIntLiteral {
  def minWidth(value: BigInt): Width = IntWidth(value.bitLength + 1)
  def apply(value: BigInt): SIntLiteral = new SIntLiteral(value, minWidth(value))
}
case class FixedLiteral(value: BigInt, width: Width, point: Width) extends Literal with UseSerializer {
  def tpe = FixedType(width, point)
  def mapExpr(f: Expression => Expression): Expression = this
  def mapType(f: Type => Type): Expression = this
  def mapWidth(f: Width => Width): Expression = FixedLiteral(value, f(width), f(point))
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = ()
  def foreachWidth(f: Width => Unit): Unit = { f(width); f(point) }
}
case class DoPrim(op: PrimOp, args: Seq[Expression], consts: Seq[BigInt], tpe: Type)
  extends Expression with UseSerializer {
  def mapExpr(f: Expression => Expression): Expression = this.copy(args = args map f)
  def mapType(f: Type => Type): Expression = this.copy(tpe = f(tpe))
  def mapWidth(f: Width => Width): Expression = this
  def foreachExpr(f: Expression => Unit): Unit = args.foreach(f)
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachWidth(f: Width => Unit): Unit = ()
}

abstract class Statement extends FirrtlNode {
  def mapStmt(f: Statement => Statement): Statement
  def mapExpr(f: Expression => Expression): Statement
  def mapType(f: Type => Type): Statement
  def mapString(f: String => String): Statement
  def mapInfo(f: Info => Info): Statement
  def foreachStmt(f: Statement => Unit): Unit
  def foreachExpr(f: Expression => Unit): Unit
  def foreachType(f: Type => Unit): Unit
  def foreachString(f: String => Unit): Unit
  def foreachInfo(f: Info => Unit): Unit
}
case class DefWire(info: Info, name: String, tpe: Type) extends Statement with IsDeclaration with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = this
  def mapType(f: Type => Type): Statement = DefWire(info, name, f(tpe))
  def mapString(f: String => String): Statement = DefWire(info, f(name), tpe)
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class DefRegister(
    info: Info,
    name: String,
    tpe: Type,
    clock: Expression,
    reset: Expression,
    init: Expression) extends Statement with IsDeclaration with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement =
    DefRegister(info, name, tpe, f(clock), f(reset), f(init))
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { f(clock); f(reset); f(init) }
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}

object DefInstance {
  def apply(name: String, module: String): DefInstance = DefInstance(NoInfo, name, module)
}

case class DefInstance(info: Info, name: String, module: String, tpe: Type = UnknownType)
    extends Statement with IsDeclaration with UseSerializer {
  def mapExpr(f: Expression => Expression): Statement = this
  def mapStmt(f: Statement => Statement): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(tpe = f(tpe))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = f(tpe)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}

object ReadUnderWrite extends Enumeration {
  val Undefined = Value("undefined")
  val Old = Value("old")
  val New = Value("new")
}

case class DefMemory(
    info: Info,
    name: String,
    dataType: Type,
    depth: BigInt,
    writeLatency: Int,
    readLatency: Int,
    readers: Seq[String],
    writers: Seq[String],
    readwriters: Seq[String],
    // TODO: handle read-under-write
    readUnderWrite: ReadUnderWrite.Value = ReadUnderWrite.Undefined)
  extends Statement with IsDeclaration with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = this
  def mapType(f: Type => Type): Statement = this.copy(dataType = f(dataType))
  def mapString(f: String => String): Statement = this.copy(name = f(name))
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = f(dataType)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class DefNode(info: Info, name: String, value: Expression) extends Statement with IsDeclaration with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = DefNode(info, name, f(value))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = DefNode(info, f(name), value)
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = f(value)
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class Conditionally(
    info: Info,
    pred: Expression,
    conseq: Statement,
    alt: Statement) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = Conditionally(info, pred, f(conseq), f(alt))
  def mapExpr(f: Expression => Expression): Statement = Conditionally(info, f(pred), conseq, alt)
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = { f(conseq); f(alt) }
  def foreachExpr(f: Expression => Unit): Unit = f(pred)
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}

object Block {
  def apply(head: Statement, tail: Statement*): Block = Block(head +: tail)
}

case class Block(stmts: Seq[Statement]) extends Statement with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = {
    val res = new scala.collection.mutable.ArrayBuffer[Statement]()
    var its = stmts.iterator :: Nil
    while (its.nonEmpty) {
      val it = its.head
      if (it.hasNext) {
        it.next() match {
          case EmptyStmt => // flatten out
          case b: Block =>
            its = b.stmts.iterator :: its
          case other =>
            res.append(f(other))
        }
      } else {
        its = its.tail
      }
    }
    Block(res.toSeq)
  }
  def mapExpr(f: Expression => Expression): Statement = this
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this
  def foreachStmt(f: Statement => Unit): Unit = stmts.foreach(f)
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = ()
}
case class PartialConnect(info: Info, loc: Expression, expr: Expression)
  extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = PartialConnect(info, f(loc), f(expr))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { f(loc); f(expr) }
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class Connect(info: Info, loc: Expression, expr: Expression)
  extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = Connect(info, f(loc), f(expr))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { f(loc); f(expr) }
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class IsInvalid(info: Info, expr: Expression) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = IsInvalid(info, f(expr))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = f(expr)
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class Attach(info: Info, exprs: Seq[Expression]) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = Attach(info, exprs map f)
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = exprs.foreach(f)
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class Stop(info: Info, ret: Int, clk: Expression, en: Expression) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = Stop(info, ret, f(clk), f(en))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { f(clk); f(en) }
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
case class Print(
    info: Info,
    string: StringLit,
    args: Seq[Expression],
    clk: Expression,
    en: Expression) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = Print(info, string, args map f, f(clk), f(en))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this.copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { args.foreach(f); f(clk); f(en) }
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}

// formal
object Formal extends Enumeration {
  val Assert = Value("assert")
  val Assume = Value("assume")
  val Cover = Value("cover")
}

case class Verification(
  op: Formal.Value,
  info: Info,
  clk: Expression,
  pred: Expression,
  en: Expression,
  msg: StringLit
) extends Statement with HasInfo with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement =
    copy(clk = f(clk), pred = f(pred), en = f(en))
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = copy(info = f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = { f(clk); f(pred); f(en); }
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
// end formal

case object EmptyStmt extends Statement with UseSerializer {
  def mapStmt(f: Statement => Statement): Statement = this
  def mapExpr(f: Expression => Expression): Statement = this
  def mapType(f: Type => Type): Statement = this
  def mapString(f: String => String): Statement = this
  def mapInfo(f: Info => Info): Statement = this
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachExpr(f: Expression => Unit): Unit = ()
  def foreachType(f: Type => Unit): Unit = ()
  def foreachString(f: String => Unit): Unit = ()
  def foreachInfo(f: Info => Unit): Unit = ()
}

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
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width max b.width)
    case _ => UnknownWidth
  }
  def min(x: Width): Width = (this, x) match {
    case (a: IntWidth, b: IntWidth) => IntWidth(a.width min b.width)
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
  def copy(width: BigInt = width) = IntWidth(width)
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
  def map(f: Constraint=>Constraint): Constraint = this
  override def reduce(): Constraint = this
  val children = Vector()
}
case class CalcBound(arg: Constraint) extends Bound {
  def serialize: String = Serializer.serialize(this)
  def map(f: Constraint=>Constraint): Constraint = f(arg)
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
    case _ => Open(value * that.value)
  }
  def min(that: IsKnown): IsKnown = if(value < that.value) this else that
  def max(that: IsKnown): IsKnown = if(value > that.value) this else that
  def neg: IsKnown = Open(-value)
  def floor: IsKnown = Open(value.setScale(0, BigDecimal.RoundingMode.FLOOR)) 
  def pow: IsKnown = if(value.isBinaryDouble) Open(BigDecimal(BigInt(1) << value.toInt)) else sys.error("Shouldn't be here")
}
case class Closed(value: BigDecimal) extends IsKnown with Bound {
  def serialize: String = Serializer.serialize(this)
  def +(that: IsKnown): IsKnown = that match {
    case Open(x) => Open(value + x)
    case Closed(x) => Closed(value + x)
  }
  def *(that: IsKnown): IsKnown = that match {
    case IsKnown(x) if value == BigInt(0) => Closed(0)
    case Open(x) => Open(value * x)
    case Closed(x) => Closed(value * x)
  }
  def min(that: IsKnown): IsKnown = if(value <= that.value) this else that
  def max(that: IsKnown): IsKnown = if(value >= that.value) this else that
  def neg: IsKnown = Closed(-value)
  def floor: IsKnown = Closed(value.setScale(0, BigDecimal.RoundingMode.FLOOR))
  def pow: IsKnown = if(value.isBinaryDouble) Closed(BigDecimal(BigInt(1) << value.toInt)) else sys.error("Shouldn't be here")
}

/** Types of [[FirrtlNode]] */
abstract class Type extends FirrtlNode {
  def mapType(f: Type => Type): Type
  def mapWidth(f: Width => Width): Type
  def foreachType(f: Type => Unit): Unit
  def foreachWidth(f: Width => Unit): Unit
}
abstract class GroundType extends Type {
  val width: Width
  def mapType(f: Type => Type): Type = this
  def foreachType(f: Type => Unit): Unit = ()
}
object GroundType {
  def unapply(ground: GroundType): Option[Width] = Some(ground.width)
}
abstract class AggregateType extends Type {
  def mapWidth(f: Width => Width): Type = this
  def foreachWidth(f: Width => Unit): Unit = ()
}
case class UIntType(width: Width) extends GroundType with UseSerializer {
  def mapWidth(f: Width => Width): Type = UIntType(f(width))
  def foreachWidth(f: Width => Unit): Unit = f(width)
}
case class SIntType(width: Width) extends GroundType with UseSerializer {
  def mapWidth(f: Width => Width): Type = SIntType(f(width))
  def foreachWidth(f: Width => Unit): Unit = f(width)
}
case class FixedType(width: Width, point: Width) extends GroundType with UseSerializer {
  def mapWidth(f: Width => Width): Type = FixedType(f(width), f(point))
  def foreachWidth(f: Width => Unit): Unit = { f(width); f(point) }
}
case class IntervalType(lower: Bound, upper: Bound, point: Width) extends GroundType {
  override def serialize: String = {
    val lowerString = lower match {
      case Open(l)      => s"(${dec2string(l)}, "
      case Closed(l)    => s"[${dec2string(l)}, "
      case UnknownBound => s"[?, "
      case _  => s"[?, "
    }
    val upperString = upper match {
      case Open(u)      => s"${dec2string(u)})"
      case Closed(u)    => s"${dec2string(u)}]"
      case UnknownBound => s"?]"
      case _  => s"?]"
    }
    val bounds = (lower, upper) match {
      case (k1: IsKnown, k2: IsKnown) => lowerString + upperString
      case _ => ""
    }
    val pointString = point match {
      case IntWidth(i)  => "." + i.toString
      case _ => ""
    }
    "Interval" + bounds + pointString
  }

  private lazy val bp = point.asInstanceOf[IntWidth].width.toInt
  private def precision: Option[BigDecimal] = point match {
    case IntWidth(width) =>
      val bp = width.toInt
      if(bp >= 0) Some(BigDecimal(1) / BigDecimal(BigInt(1) << bp)) else Some(BigDecimal(BigInt(1) << -bp))
    case other => None
  }

  def min: Option[BigDecimal] = (lower, precision) match {
    case (Open(a), Some(prec))  => a / prec match {
      case x if trim(x).isWhole => Some(a + prec) // add precision for open lower bound i.e. (-4 -> [3 for bp = 0
      case x => Some(x.setScale(0, CEILING) * prec) // Deal with unrepresentable bound representations (finite BP) -- new closed form l > original l
    }
    case (Closed(a), Some(prec)) => Some((a / prec).setScale(0, CEILING) * prec)
    case other => None
  }

  def max: Option[BigDecimal] = (upper, precision) match {
    case (Open(a), Some(prec)) => a / prec match {
      case x if trim(x).isWhole => Some(a - prec) // subtract precision for open upper bound
      case x => Some(x.setScale(0, FLOOR) * prec)
    }
    case (Closed(a), Some(prec)) => Some((a / prec).setScale(0, FLOOR) * prec)
  }

  def minAdjusted: Option[BigInt] = min.map(_ * BigDecimal(BigInt(1) << bp) match {
    case x if trim(x).isWhole | x.doubleValue == 0.0 => x.toBigInt
    case x => sys.error(s"MinAdjusted should be a whole number: $x. Min is $min. BP is $bp. Precision is $precision. Lower is ${lower}.")
  })

  def maxAdjusted: Option[BigInt] = max.map(_ * BigDecimal(BigInt(1) << bp) match {
    case x if trim(x).isWhole => x.toBigInt
    case x => sys.error(s"MaxAdjusted should be a whole number: $x")
  })

  /** If bounds are known, calculates the width, otherwise returns UnknownWidth */
  lazy val width: Width = (point, lower, upper) match {
    case (IntWidth(i), l: IsKnown, u: IsKnown) =>
      IntWidth(Math.max(Utils.getSIntWidth(minAdjusted.get), Utils.getSIntWidth(maxAdjusted.get)))
    case _ => UnknownWidth
  }

  /** If bounds are known, returns a sequence of all possible values inside this interval */
  lazy val range: Option[Seq[BigDecimal]] = (lower, upper, point) match {
    case (l: IsKnown, u: IsKnown, p: IntWidth) =>
      if(min.get > max.get) Some(Nil) else Some(Range.BigDecimal(min.get, max.get, precision.get))
    case _ => None
  }

  override def mapWidth(f: Width => Width): Type = this.copy(point = f(point))
  override def foreachWidth(f: Width => Unit): Unit = f(point)
}

case class BundleType(fields: Seq[Field]) extends AggregateType with UseSerializer {
  def mapType(f: Type => Type): Type =
    BundleType(fields map (x => x.copy(tpe = f(x.tpe))))
  def foreachType(f: Type => Unit): Unit = fields.foreach{ x => f(x.tpe) }
}
case class VectorType(tpe: Type, size: Int) extends AggregateType with UseSerializer {
  def mapType(f: Type => Type): Type = this.copy(tpe = f(tpe))
  def foreachType(f: Type => Unit): Unit = f(tpe)
}
case object ClockType extends GroundType with UseSerializer {
  val width = IntWidth(1)
  def mapWidth(f: Width => Width): Type = this
  def foreachWidth(f: Width => Unit): Unit = ()
}
/* Abstract reset, will be inferred to UInt<1> or AsyncReset */
case object ResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
  def mapWidth(f: Width => Width): Type = this
  def foreachWidth(f: Width => Unit): Unit = ()
}
case object AsyncResetType extends GroundType with UseSerializer {
  val width = IntWidth(1)
  def mapWidth(f: Width => Width): Type = this
  def foreachWidth(f: Width => Unit): Unit = ()
}
case class AnalogType(width: Width) extends GroundType with UseSerializer {
  def mapWidth(f: Width => Width): Type = AnalogType(f(width))
  def foreachWidth(f: Width => Unit): Unit = f(width)
}
case object UnknownType extends Type with UseSerializer {
  def mapType(f: Type => Type): Type = this
  def mapWidth(f: Width => Width): Type = this
  def foreachType(f: Type => Unit): Unit = ()
  def foreachWidth(f: Width => Unit): Unit = ()
}

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
    info: Info,
    name: String,
    direction: Direction,
    tpe: Type) extends FirrtlNode with IsDeclaration with UseSerializer {
  def mapType(f: Type => Type): Port = Port(info, name, direction, f(tpe))
  def mapString(f: String => String): Port = Port(info, f(name), direction, tpe)
}

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
  val info : Info
  val name : String
  val ports : Seq[Port]
  def mapStmt(f: Statement => Statement): DefModule
  def mapPort(f: Port => Port): DefModule
  def mapString(f: String => String): DefModule
  def mapInfo(f: Info => Info): DefModule
  def foreachStmt(f: Statement => Unit): Unit
  def foreachPort(f: Port => Unit): Unit
  def foreachString(f: String => Unit): Unit
  def foreachInfo(f: Info => Unit): Unit
}
/** Internal Module
  *
  * An instantiable hardware block
  */
case class Module(info: Info, name: String, ports: Seq[Port], body: Statement) extends DefModule with UseSerializer {
  def mapStmt(f: Statement => Statement): DefModule = this.copy(body = f(body))
  def mapPort(f: Port => Port): DefModule = this.copy(ports = ports map f)
  def mapString(f: String => String): DefModule = this.copy(name = f(name))
  def mapInfo(f: Info => Info): DefModule = this.copy(f(info))
  def foreachStmt(f: Statement => Unit): Unit = f(body)
  def foreachPort(f: Port => Unit): Unit = ports.foreach(f)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
/** External Module
  *
  * Generally used for Verilog black boxes
  * @param defname Defined name of the external module (ie. the name Firrtl will emit)
  */
case class ExtModule(
    info: Info,
    name: String,
    ports: Seq[Port],
    defname: String,
    params: Seq[Param]) extends DefModule with UseSerializer {
  def mapStmt(f: Statement => Statement): DefModule = this
  def mapPort(f: Port => Port): DefModule = this.copy(ports = ports map f)
  def mapString(f: String => String): DefModule = this.copy(name = f(name))
  def mapInfo(f: Info => Info): DefModule = this.copy(f(info))
  def foreachStmt(f: Statement => Unit): Unit = ()
  def foreachPort(f: Port => Unit): Unit = ports.foreach(f)
  def foreachString(f: String => Unit): Unit = f(name)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}

case class Circuit(info: Info, modules: Seq[DefModule], main: String)
  extends FirrtlNode with HasInfo with UseSerializer {
  def mapModule(f: DefModule => DefModule): Circuit = this.copy(modules = modules map f)
  def mapString(f: String => String): Circuit = this.copy(main = f(main))
  def mapInfo(f: Info => Info): Circuit = this.copy(f(info))
  def foreachModule(f: DefModule => Unit): Unit = modules foreach f
  def foreachString(f: String => Unit): Unit = f(main)
  def foreachInfo(f: Info => Unit): Unit = f(info)
}
