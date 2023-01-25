// SPDX-License-Identifier: Apache-2.0

package firrtl.ir

import firrtl.Utils
import firrtl.backends.experimental.smt.random.DefRandom
import firrtl.constraint.Constraint

case class Version(major: Int, minor: Int, patch: Int) {
  def serialize: String = s"$major.$minor.$patch"
  def incompatible(that: Version): Boolean =
    this.major > that.major || (this.major == that.major && this.minor > that.minor)
}

object Serializer {
  val NewLine = '\n'
  val Indent = "  "

  // The version supported by the serializer.
  val version = Version(1, 1, 0)

  /** Converts a `FirrtlNode` into its string representation with
    * default indentation.
    */
  def serialize(node: FirrtlNode): String = {
    serialize(node, 0)
  }

  /** Converts a `FirrtlNode` into its string representation. */
  def serialize(node: FirrtlNode, indent: Int): String = {
    val builder = new StringBuilder()
    node match {
      case n: Info        => s(n)(builder, indent)
      case n: StringLit   => s(n)(builder, indent)
      case n: Expression  => s(n)(builder, indent)
      case n: Statement   => builder ++= lazily(n, indent).mkString
      case n: Width       => s(n)(builder, indent)
      case n: Orientation => s(n)(builder, indent)
      case n: Field       => s(n)(builder, indent)
      case n: Type        => s(n)(builder, indent)
      case n: Direction   => s(n)(builder, indent)
      case n: Port        => s(n)(builder, indent)
      case n: Param       => s(n)(builder, indent)
      case n: DefModule   => builder ++= lazily(n, indent).mkString
      case n: Circuit     => builder ++= lazily(n, indent).mkString
      case other => builder ++= other.serialize // Handle user-defined nodes
    }
    builder.toString()
  }

  /** Converts a `FirrtlNode` to an Iterable of Strings
    *
    * The Strings in the Iterable can be concatenated to give the String representation of the
    * `FirrtlNode`. This is useful for buffered emission, especially for large Circuits that
    * encroach on the JVM limit on String size (2 GiB).
    */
  def lazily(node: FirrtlNode): Iterable[String] = lazily(node, 0)

  /** Converts a `FirrtlNode` to an Iterable of Strings
    *
    * The Strings in the Iterable can be concatenated to give the String representation of the
    * `FirrtlNode`. This is useful for buffered emission, especially for large Circuits that
    * encroach on the JVM limit on String size (2 GiB).
    */
  def lazily(node: FirrtlNode, indent: Int): Iterable[String] = new Iterable[String] {
    def iterator = node match {
      case n: Statement => sIt(n)(indent)
      case n: DefModule => sIt(n)(indent)
      case n: Circuit   => sIt(n)(indent)
      case other => Iterator(serialize(other, indent))
    }
  }

  /** Converts a `Constraint` into its string representation. */
  def serialize(con: Constraint): String = {
    val builder = new StringBuilder()
    s(con)(builder)
    builder.toString()
  }

  private def flattenInfo(infos: Seq[Info]): Seq[FileInfo] = infos.flatMap {
    case NoInfo => Seq()
    case f: FileInfo => Seq(f)
    case MultiInfo(infos) => flattenInfo(infos)
  }

  private def s(node: Info)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case f: FileInfo => b ++= " @["; b ++= f.escaped; b ++= "]"
    case NoInfo => // empty string
    case m: MultiInfo =>
      val infos = m.flatten
      if (infos.nonEmpty) {
        val lastId = infos.length - 1
        b ++= " @["
        infos.zipWithIndex.foreach { case (f, i) => b ++= f.escaped; if (i < lastId) b += ' ' }
        b += ']'
      }
    case other => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(str: StringLit)(implicit b: StringBuilder, indent: Int): Unit = b ++= str.serialize

  private def s(node: Expression)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Reference(name, _, _, _) => b ++= name
    case DoPrim(op, args, consts, _) =>
      b ++= op.toString; b += '('; s(args, ", ", consts.isEmpty); s(consts, ", "); b += ')'
    case UIntLiteral(value, width) =>
      b ++= "UInt"; s(width); b ++= "(\"h"; b ++= value.toString(16); b ++= "\")"
    case SubField(expr, name, _, _)   => s(expr); b += '.'; b ++= name
    case SubIndex(expr, value, _, _)  => s(expr); b += '['; b ++= value.toString; b += ']'
    case SubAccess(expr, index, _, _) => s(expr); b += '['; s(index); b += ']'
    case Mux(cond, tval, fval, _) =>
      b ++= "mux("; s(cond); b ++= ", "; s(tval); b ++= ", "; s(fval); b += ')'
    case ValidIf(cond, value, _) => b ++= "validif("; s(cond); b ++= ", "; s(value); b += ')'
    case SIntLiteral(value, width) =>
      b ++= "SInt"; s(width); b ++= "(\"h"; b ++= value.toString(16); b ++= "\")"
    case FixedLiteral(value, width, point) =>
      b ++= "Fixed"; s(width); sPoint(point)
      b ++= "(\"h"; b ++= value.toString(16); b ++= "\")"
    // WIR
    case firrtl.WVoid           => b ++= "VOID"
    case firrtl.WInvalid        => b ++= "INVALID"
    case firrtl.EmptyExpression => b ++= "EMPTY"
    case other                  => b ++= other.serialize // Handle user-defined nodes
  }

  // Helper for some not-real Statements that only exist for Serialization
  private abstract class PseudoStatement extends Statement {
    def foreachExpr(f:   Expression => Unit):       Unit = ???
    def foreachInfo(f:   Info => Unit):             Unit = ???
    def foreachStmt(f:   Statement => Unit):        Unit = ???
    def foreachString(f: String => Unit):           Unit = ???
    def foreachType(f:   Type => Unit):             Unit = ???
    def mapExpr(f:       Expression => Expression): Statement = ???
    def mapInfo(f:       Info => Info):             Statement = ???
    def mapStmt(f:       Statement => Statement):   Statement = ???
    def mapString(f:     String => String):         Statement = ???
    def mapType(f:       Type => Type):             Statement = ???
    def serialize: String = ???
  }

  // To treat Statments as Iterable, we need to flatten out when scoping
  private case class WhenBegin(info: Info, pred: Expression) extends PseudoStatement
  private case object AltBegin extends PseudoStatement
  private case object WhenEnd extends PseudoStatement

  // This does not extend Iterator[Statement] because
  //  1. It is extended by StmtsSerializer which extends Iterator[String]
  //  2. Flattening out whens introduces fake Statements needed for [un]indenting
  private abstract class FlatStmtsIterator(stmts: Iterable[Statement]) {
    private var underlying: Iterator[Statement] = stmts.iterator

    protected def hasNextStmt = underlying.hasNext

    protected def nextStmt(): Statement = {
      var next: Statement = null
      while (next == null && hasNextStmt) {
        val head = underlying
        head.next() match {
          case b: Block if b.stmts.isEmpty =>
            next = EmptyStmt
          case b: Block =>
            val first = b.stmts.iterator
            val last = underlying
            underlying = first ++ last
          case Conditionally(info, pred, conseq, alt) =>
            val begin = WhenBegin(info, pred)
            val stmts = if (alt == EmptyStmt) {
              Iterator(begin, conseq, WhenEnd)
            } else {
              Iterator(begin, conseq, AltBegin, alt, WhenEnd)
            }
            val last = underlying
            underlying = stmts ++ last
          case other =>
            next = other
        }
      }
      next
    }
  }

  // Extend FlatStmtsIterator directly (rather than wrapping a FlatStmtsIterator object) to reduce
  // the boxing overhead
  private class StmtsSerializer(stmts: Iterable[Statement], initialIndent: Int)
      extends FlatStmtsIterator(stmts)
      with Iterator[String] {

    private def bufferSize = 2048

    // We could initialze the StringBuilder size, but this is bad for small modules which may not
    // even reach the bufferSize.
    private implicit val b = new StringBuilder

    // The flattening of Whens into WhenBegin and friends requires us to keep track of the
    // indention level
    private implicit var indent: Int = initialIndent

    def hasNext: Boolean = this.hasNextStmt

    def next(): String = {
      def consumeStmt(stmt: Statement): Unit = {
        stmt match {
          case wb: WhenBegin =>
            doIndent()
            b ++= "when "; s(wb.pred); b ++= " :"; s(wb.info)
            indent += 1
          case AltBegin =>
            indent -= 1
            doIndent()
            b ++= "else :"
            indent += 1
          case WhenEnd =>
            indent -= 1
          case other =>
            doIndent()
            s(other)
        }
        if (this.hasNext && stmt != WhenEnd) {
          newLineNoIndent()
        }
      }
      b.clear()
      // There must always be at least 1 Statement because we're nonEmpty
      var stmt: Statement = nextStmt()
      while (stmt != null && b.size < bufferSize) {
        consumeStmt(stmt)
        stmt = nextStmt()
      }
      if (stmt != null) {
        consumeStmt(stmt)
      }
      b.toString
    }
  }

  private def sIt(node: Statement)(implicit indent: Int): Iterator[String] = node match {
    case b: Block =>
      if (b.stmts.isEmpty) sIt(EmptyStmt)
      else new StmtsSerializer(b.stmts, indent)
    case cond: Conditionally => new StmtsSerializer(Seq(cond), indent)
    case other =>
      implicit val b = new StringBuilder
      doIndent()
      s(other)
      Iterator(b.toString)
  }

  private def s(node: Statement)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case DefNode(info, name, value) => b ++= "node "; b ++= name; b ++= " = "; s(value); s(info)
    case Connect(info, loc, expr)   => s(loc); b ++= " <= "; s(expr); s(info)
    case c: Conditionally => b ++= sIt(c).mkString
    case EmptyStmt => b ++= "skip"
    case bb: Block => b ++= sIt(bb).mkString
    case stop @ Stop(info, ret, clk, en) =>
      b ++= "stop("; s(clk); b ++= ", "; s(en); b ++= ", "; b ++= ret.toString; b += ')'
      sStmtName(stop.name); s(info)
    case print @ Print(info, string, args, clk, en) =>
      b ++= "printf("; s(clk); b ++= ", "; s(en); b ++= ", "; b ++= string.escape
      if (args.nonEmpty) b ++= ", "; s(args, ", "); b += ')'
      sStmtName(print.name); s(info)
    case IsInvalid(info, expr)    => s(expr); b ++= " is invalid"; s(info)
    case DefWire(info, name, tpe) => b ++= "wire "; b ++= name; b ++= " : "; s(tpe); s(info)
    case DefRegister(info, name, tpe, clock, reset, init) =>
      b ++= "reg "; b ++= name; b ++= " : "; s(tpe); b ++= ", "; s(clock); b ++= " with :"; newLineAndIndent(1)
      b ++= "reset => ("; s(reset); b ++= ", "; s(init); b += ')'; s(info)
    case DefRandom(info, name, tpe, clock, en) =>
      b ++= "rand "; b ++= name; b ++= " : "; s(tpe);
      if (clock.isDefined) { b ++= ", "; s(clock.get); }
      en match { case Utils.True() => case _ => b ++= " when "; s(en) }
      s(info)
    case DefInstance(info, name, module, _) => b ++= "inst "; b ++= name; b ++= " of "; b ++= module; s(info)
    case DefMemory(
          info,
          name,
          dataType,
          depth,
          writeLatency,
          readLatency,
          readers,
          writers,
          readwriters,
          readUnderWrite
        ) =>
      b ++= "mem "; b ++= name; b ++= " :"; s(info); newLineAndIndent(1)
      b ++= "data-type => "; s(dataType); newLineAndIndent(1)
      b ++= "depth => "; b ++= depth.toString(); newLineAndIndent(1)
      b ++= "read-latency => "; b ++= readLatency.toString; newLineAndIndent(1)
      b ++= "write-latency => "; b ++= writeLatency.toString; newLineAndIndent(1)
      readers.foreach { r => b ++= "reader => "; b ++= r; newLineAndIndent(1) }
      writers.foreach { w => b ++= "writer => "; b ++= w; newLineAndIndent(1) }
      readwriters.foreach { r => b ++= "readwriter => "; b ++= r; newLineAndIndent(1) }
      b ++= "read-under-write => "; b ++= readUnderWrite.toString
    case PartialConnect(info, loc, expr) => s(loc); b ++= " <- "; s(expr); s(info)
    case Attach(info, exprs)             =>
      // exprs should never be empty since the attach statement takes *at least* two signals according to the spec
      b ++= "attach ("; s(exprs, ", "); b += ')'; s(info)
    case veri @ Verification(op, info, clk, pred, en, msg) =>
      b ++= op.toString; b += '('; s(List(clk, pred, en), ", ", false); b ++= msg.escape
      b += ')'; sStmtName(veri.name); s(info)

    // WIR
    case firrtl.CDefMemory(info, name, tpe, size, seq, readUnderWrite) =>
      if (seq) b ++= "smem " else b ++= "cmem "
      b ++= name; b ++= " : "; s(tpe); b ++= " ["; b ++= size.toString(); b += ']'
      b += ' '; b ++= readUnderWrite.toString; s(info)
    case firrtl.CDefMPort(info, name, _, mem, exps, direction) =>
      b ++= direction.serialize; b ++= " mport "; b ++= name; b ++= " = "; b ++= mem
      b += '['; s(exps.head); b ++= "], "; s(exps(1)); s(info)
    case firrtl.WDefInstanceConnector(info, name, module, tpe, portCons) =>
      b ++= "inst "; b ++= name; b ++= " of "; b ++= module; b ++= " with "; s(tpe); b ++= " connected to ("
      s(portCons.map(_._2), ",  "); b += ')'; s(info)
    case other => b ++= other.serialize // Handle user-defined nodes
  }

  private def sStmtName(lbl: String)(implicit b: StringBuilder): Unit = {
    if (lbl.nonEmpty) { b ++= s" : $lbl" }
  }

  private def s(node: Width)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case IntWidth(width) => b += '<'; b ++= width.toString(); b += '>'
    case UnknownWidth    => // empty string
    case CalcWidth(arg)  => b ++= "calcw("; s(arg); b += ')'
    case VarWidth(name)  => b += '<'; b ++= name; b += '>'
    case other           => b ++= other.serialize // Handle user-defined nodes
  }

  private def sPoint(node: Width)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case IntWidth(width) => b ++= "<<"; b ++= width.toString(); b ++= ">>"
    case UnknownWidth    => // empty string
    case CalcWidth(arg)  => b ++= "calcw("; s(arg); b += ')'
    case VarWidth(name)  => b ++= "<<"; b ++= name; b ++= ">>"
    case other           => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(node: Orientation)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Default => // empty string
    case Flip    => b ++= "flip "
    case other   => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(node: Field)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Field(name, flip, tpe) => s(flip); b ++= name; b ++= " : "; s(tpe)
  }

  private def s(node: Type)(implicit b: StringBuilder, indent: Int): Unit = node match {
    // Types
    case UIntType(width: Width) => b ++= "UInt"; s(width)
    case SIntType(width: Width) => b ++= "SInt"; s(width)
    case FixedType(width, point) => b ++= "Fixed"; s(width); sPoint(point)
    case BundleType(fields)      => b ++= "{ "; sField(fields, ", "); b += '}'
    case VectorType(tpe, size)   => s(tpe); b += '['; b ++= size.toString; b += ']'
    case ClockType               => b ++= "Clock"
    case ResetType               => b ++= "Reset"
    case AsyncResetType          => b ++= "AsyncReset"
    case AnalogType(width)       => b ++= "Analog"; s(width)
    case UnknownType             => b += '?'
    // the IntervalType has a complicated custom serialization method which does not recurse
    case i: IntervalType => b ++= i.serialize
    case other => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(node: Direction)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Input  => b ++= "input"
    case Output => b ++= "output"
    case other  => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(node: Port)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Port(info, name, direction, tpe) =>
      s(direction); b += ' '; b ++= name; b ++= " : "; s(tpe); s(info)
  }

  private def s(node: Param)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case IntParam(name, value)    => b ++= "parameter "; b ++= name; b ++= " = "; b ++= value.toString
    case DoubleParam(name, value) => b ++= "parameter "; b ++= name; b ++= " = "; b ++= value.toString
    case StringParam(name, value) => b ++= "parameter "; b ++= name; b ++= " = "; b ++= value.escape
    case RawStringParam(name, value) =>
      b ++= "parameter "; b ++= name; b ++= " = "
      b += '\''; b ++= value.replace("'", "\\'"); b += '\''
    case other => b ++= other.serialize // Handle user-defined nodes
  }

  private def sIt(node: DefModule)(implicit indent: Int): Iterator[String] = node match {
    case Module(info, name, ports, body) =>
      val start = {
        implicit val b = new StringBuilder
        doIndent(0); b ++= "module "; b ++= name; b ++= " :"; s(info)
        ports.foreach { p => newLineAndIndent(1); s(p) }
        newLineNoIndent() // add a blank line between port declaration and body
        newLineNoIndent() // newline for body, sIt will indent
        b.toString
      }
      Iterator(start) ++ sIt(body)(indent + 1)
    case ExtModule(info, name, ports, defname, params) =>
      implicit val b = new StringBuilder
      doIndent(0); b ++= "extmodule "; b ++= name; b ++= " :"; s(info)
      ports.foreach { p => newLineAndIndent(1); s(p) }
      newLineAndIndent(1); b ++= "defname = "; b ++= defname
      params.foreach { p => newLineAndIndent(1); s(p) }
      Iterator(b.toString)
    case other =>
      Iterator(Indent * indent, other.serialize) // Handle user-defined nodes
  }

  private def sIt(node: Circuit)(implicit indent: Int): Iterator[String] = node match {
    case Circuit(info, modules, main) =>
      val prelude = {
        implicit val b = new StringBuilder // Scope this so we don't accidentally pass it anywhere
        b ++= s"FIRRTL version ${version.serialize}\n"
        b ++= "circuit "; b ++= main; b ++= " :"; s(info)
        b.toString
      }
      Iterator(prelude) ++
        modules.iterator.zipWithIndex.flatMap {
          case (m, i) =>
            val newline = Iterator(if (i == 0) s"$NewLine" else s"${NewLine}${NewLine}")
            newline ++ sIt(m)(indent + 1)
        } ++
        Iterator(s"$NewLine")
  }

  // serialize constraints
  private def s(const: Constraint)(implicit b: StringBuilder): Unit = const match {
    // Bounds
    case UnknownBound   => b += '?'
    case CalcBound(arg) => b ++= "calcb("; s(arg); b += ')'
    case VarBound(name) => b ++= name
    case Open(value)    => b ++ "o("; b ++= value.toString; b += ')'
    case Closed(value)  => b ++ "c("; b ++= value.toString; b += ')'
    case other          => b ++= other.serialize // Handle user-defined nodes
  }

  /** create a new line with the appropriate indent */
  private def newLineAndIndent(inc: Int = 0)(implicit b: StringBuilder, indent: Int): Unit = {
    b += NewLine; doIndent(inc)
  }

  private def newLineNoIndent()(implicit b: StringBuilder): Unit = b += NewLine

  /** create indent, inc allows for a temporary increment */
  private def doIndent(inc: Int = 0)(implicit b: StringBuilder, indent: Int): Unit = {
    (0 until (indent + inc)).foreach { _ => b ++= Indent }
  }

  /** serialize firrtl Expression nodes with a custom separator and the option to include the separator at the end */
  private def s(
    nodes:      Iterable[Expression],
    sep:        String,
    noFinalSep: Boolean = true
  )(
    implicit b: StringBuilder,
    indent:     Int
  ): Unit = {
    val it = nodes.iterator
    while (it.hasNext) {
      s(it.next())
      if (!noFinalSep || it.hasNext) b ++= sep
    }
  }

  /** serialize firrtl Field nodes with a custom separator and the option to include the separator at the end */
  @inline
  private def sField(nodes: Iterable[Field], sep: String)(implicit b: StringBuilder, indent: Int): Unit = {
    val it = nodes.iterator
    while (it.hasNext) {
      s(it.next())
      if (it.hasNext) b ++= sep
    }
  }

  /** serialize BigInts with a custom separator */
  private def s(consts: Iterable[BigInt], sep: String)(implicit b: StringBuilder): Unit = {
    val it = consts.iterator
    while (it.hasNext) {
      b ++= it.next().toString()
      if (it.hasNext) b ++= sep
    }
  }
}
