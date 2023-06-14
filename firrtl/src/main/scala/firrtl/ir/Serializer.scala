// SPDX-License-Identifier: Apache-2.0

package firrtl.ir

import firrtl.annotations.{Annotation, JsonProtocol}

case class Version(major: Int, minor: Int, patch: Int) {
  def serialize: String = s"$major.$minor.$patch"
  def incompatible(that: Version): Boolean =
    this.major > that.major || (this.major == that.major && this.minor > that.minor)
}

object Serializer {
  val NewLine = '\n'
  val Indent = "  "

  // The version supported by the serializer.
  val version = Version(2, 0, 0)

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
      case n: Info             => s(n)(builder, indent)
      case n: StringLit        => s(n)(builder, indent)
      case n: Expression       => s(n)(builder, indent)
      case n: Statement        => builder ++= lazily(n, indent).mkString
      case n: Width            => s(n)(builder, indent)
      case n: Orientation      => s(n)(builder, indent)
      case n: Field            => s(n)(builder, indent)
      case n: Type             => s(n)(builder, indent)
      case n: Direction        => s(n)(builder, indent)
      case n: Port             => s(n)(builder, indent)
      case n: Param            => s(n)(builder, indent)
      case n: DefModule        => builder ++= lazily(n, indent).mkString
      case n: Circuit          => builder ++= lazily(n, indent).mkString
      case n: CircuitWithAnnos => builder ++= lazily(n, indent).mkString
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
      case n: Statement        => sIt(n)(indent)
      case n: DefModule        => sIt(n)(indent)
      case n: Circuit          => sIt(n)(indent)
      case n: CircuitWithAnnos => sIt(n)(indent)
      case other => Iterator(serialize(other, indent))
    }
  }.view // TODO replace .view with constructing a view directly above, but must drop 2.12 first.

  private def flattenInfo(infos: Seq[Info]): Seq[FileInfo] = infos.flatMap {
    case NoInfo => Seq()
    case f: FileInfo => Seq(f)
  }

  private def s(node: Info)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case f: FileInfo => b ++= " @["; b ++= f.escaped; b ++= "]"
    case NoInfo => // empty string
    case other  => b ++= other.serialize // Handle user-defined nodes
  }

  private def s(str: StringLit)(implicit b: StringBuilder, indent: Int): Unit = b ++= str.serialize

  private def s(node: Expression)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case Reference(name, _) => b ++= name
    case DoPrim(op, args, consts, _) =>
      b ++= op.toString; b += '('; s(args, ", ", consts.isEmpty); s(consts, ", "); b += ')'
    case UIntLiteral(value, width) =>
      b ++= "UInt"; s(width); b ++= "(\"h"; b ++= value.toString(16); b ++= "\")"
    case SubField(expr, name, _)   => s(expr); b += '.'; b ++= name
    case SubIndex(expr, value, _)  => s(expr); b += '['; b ++= value.toString; b += ']'
    case SubAccess(expr, index, _) => s(expr); b += '['; s(index); b += ']'
    case Mux(cond, tval, fval, _) =>
      b ++= "mux("; s(cond); b ++= ", "; s(tval); b ++= ", "; s(fval); b += ')'
    case ValidIf(cond, value, _) => b ++= "validif("; s(cond); b ++= ", "; s(value); b += ')'
    case SIntLiteral(value, width) =>
      b ++= "SInt"; s(width); b ++= "(\"h"; b ++= value.toString(16); b ++= "\")"
    case ProbeExpr(expr, _)   => b ++= "probe("; s(expr); b += ')'
    case RWProbeExpr(expr, _) => b ++= "rwprobe("; s(expr); b += ')'
    case ProbeRead(expr, _)   => b ++= "read("; s(expr); b += ')'
    case other                => b ++= other.serialize // Handle user-defined nodes
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
    case DefRegister(info, name, tpe, clock) =>
      b ++= "reg "; b ++= name; b ++= " : "; s(tpe); b ++= ", "; s(clock); s(info)
    case DefRegisterWithReset(info, name, tpe, clock, reset, init) =>
      b ++= "reg "; b ++= name; b ++= " : "; s(tpe); b ++= ", "; s(clock); b ++= " with :"; newLineAndIndent(1)
      b ++= "reset => ("; s(reset); b ++= ", "; s(init); b += ')'; s(info)
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
    case Attach(info, exprs) =>
      // exprs should never be empty since the attach statement takes *at least* two signals according to the spec
      b ++= "attach ("; s(exprs, ", "); b += ')'; s(info)
    case veri @ Verification(op, info, clk, pred, en, msg) =>
      b ++= op.toString; b += '('; s(List(clk, pred, en), ", ", false); b ++= msg.escape
      b += ')'; sStmtName(veri.name); s(info)

    // WIR
    case firrtl.CDefMemory(info, name, tpe, size, seq, readUnderWrite) =>
      if (seq) b ++= "smem " else b ++= "cmem "
      b ++= name; b ++= " : "; s(tpe); b ++= " ["; b ++= size.toString(); b += ']'
      if (readUnderWrite != ReadUnderWrite.Undefined) { // undefined is the default
        b += ' '; b ++= readUnderWrite.toString
      }
      s(info)
    case firrtl.CDefMPort(info, name, _, mem, exps, direction) =>
      b ++= direction.serialize; b ++= " mport "; b ++= name; b ++= " = "; b ++= mem
      b += '['; s(exps.head); b ++= "], "; s(exps(1)); s(info)
    case ProbeDefine(info, sink, probeExpr) =>
      b ++= "define "; s(sink); b ++= " = "; s(probeExpr); s(info)
    case ProbeForceInitial(info, probe, value) =>
      b ++= "force_initial("; s(probe); b ++= ", "; s(value); b += ')'; s(info)
    case ProbeReleaseInitial(info, probe) =>
      b ++= "release_initial("; s(probe); b += ')'; s(info)
    case ProbeForce(info, clock, cond, probe, value) =>
      b ++= "force("; s(clock); b ++= ", "; s(cond); b ++= ", "; s(probe); b ++= ", "; s(value); b += ')'; s(info)
    case ProbeRelease(info, clock, cond, probe) =>
      b ++= "release("; s(clock); b ++= ", "; s(cond); b ++= ", "; s(probe); b += ')'; s(info)
    case other => b ++= other.serialize // Handle user-defined nodes
  }

  private def sStmtName(lbl: String)(implicit b: StringBuilder): Unit = {
    if (lbl.nonEmpty) { b ++= s" : $lbl" }
  }

  private def s(node: Width)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case IntWidth(width) => b += '<'; b ++= width.toString(); b += '>'
    case UnknownWidth    => // empty string
    case other           => b ++= other.serialize // Handle user-defined nodes
  }

  private def sPoint(node: Width)(implicit b: StringBuilder, indent: Int): Unit = node match {
    case IntWidth(width) => b ++= "<<"; b ++= width.toString(); b ++= ">>"
    case UnknownWidth    => // empty string
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
    case ProbeType(underlying: Type) => b ++= "Probe<"; s(underlying); b += '>'
    case RWProbeType(underlying: Type) => b ++= "RWProbe<"; s(underlying); b += '>'
    case ConstType(underlying: Type) => b ++= "const "; s(underlying)
    case UIntType(width: Width) => b ++= "UInt"; s(width)
    case SIntType(width: Width) => b ++= "SInt"; s(width)
    case BundleType(fields)    => b ++= "{ "; sField(fields, ", "); b += '}'
    case VectorType(tpe, size) => s(tpe); b += '['; b ++= size.toString; b += ']'
    case ClockType             => b ++= "Clock"
    case ResetType             => b ++= "Reset"
    case AsyncResetType        => b ++= "AsyncReset"
    case AnalogType(width)     => b ++= "Analog"; s(width)
    case UnknownType           => b += '?'
    case other                 => b ++= other.serialize // Handle user-defined nodes
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
    case IntModule(info, name, ports, intrinsic, params) =>
      implicit val b = new StringBuilder
      doIndent(0); b ++= "intmodule "; b ++= name; b ++= " :"; s(info)
      ports.foreach { p => newLineAndIndent(1); s(p) }
      newLineAndIndent(1); b ++= "intrinsic = "; b ++= intrinsic
      params.foreach { p => newLineAndIndent(1); s(p) }
      Iterator(b.toString)
    case other =>
      Iterator(Indent * indent, other.serialize) // Handle user-defined nodes
  }

  private def sIt(node: Circuit)(implicit indent: Int): Iterator[String] =
    sIt(CircuitWithAnnos(node, Nil))

  // TODO make Annotation serialization lazy
  private def sIt(node: CircuitWithAnnos)(implicit indent: Int): Iterator[String] = {
    val CircuitWithAnnos(circuit, annotations) = node
    val prelude = {
      implicit val b = new StringBuilder
      b ++= s"FIRRTL version ${version.serialize}\n"
      b ++= "circuit "; b ++= circuit.main; b ++= " :";
      if (annotations.nonEmpty) {
        b ++= "%["; b ++= JsonProtocol.serialize(annotations); b ++= "]";
      }
      s(circuit.info)
      Iterator(b.toString)
    }
    prelude ++
      circuit.modules.iterator.zipWithIndex.flatMap {
        case (m, i) =>
          val newline = Iterator(if (i == 0) s"$NewLine" else s"${NewLine}${NewLine}")
          newline ++ sIt(m)(indent + 1)
      } ++
      Iterator(s"$NewLine")
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
