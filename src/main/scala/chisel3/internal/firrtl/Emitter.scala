// See LICENSE for license details.

package chisel3.internal.firrtl
import chisel3._
import chisel3.internal.sourceinfo.{NoSourceInfo, SourceLine}

private[chisel3] object Emitter {
  def emit(circuit: Circuit): String = new Emitter(circuit).toString
}

private class Emitter(circuit: Circuit) {
  override def toString: String = res.toString

  private def emitPort(e: Port): String =
    s"${e.dir} ${e.id.getRef.name} : ${e.id.toType}"
  private def emit(e: Command, ctx: Component): String = {
    val firrtlLine = e match {
      case e: DefPrim[_] => s"node ${e.name} = ${e.op.name}(${e.args.map(_.fullName(ctx)).mkString(", ")})"
      case e: DefWire => s"wire ${e.name} : ${e.id.toType}"
      case e: DefReg => s"reg ${e.name} : ${e.id.toType}, ${e.clock.fullName(ctx)}"
      case e: DefRegInit => s"reg ${e.name} : ${e.id.toType}, ${e.clock.fullName(ctx)} with : (reset => (${e.reset.fullName(ctx)}, ${e.init.fullName(ctx)}))"
      case e: DefMemory => s"cmem ${e.name} : ${e.t.toType}[${e.size}]"
      case e: DefSeqMemory => s"smem ${e.name} : ${e.t.toType}[${e.size}]"
      case e: DefMemPort[_] => s"${e.dir} mport ${e.name} = ${e.source.fullName(ctx)}[${e.index.fullName(ctx)}], ${e.clock.fullName(ctx)}"
      case e: Connect => s"${e.loc.fullName(ctx)} <= ${e.exp.fullName(ctx)}"
      case e: BulkConnect => s"${e.loc1.fullName(ctx)} <- ${e.loc2.fullName(ctx)}"
      case e: Stop => s"stop(${e.clock.fullName(ctx)}, UInt<1>(1), ${e.ret})"
      case e: Printf =>
        val (fmt, args) = e.pable.unpack(ctx)
        val printfArgs = Seq(e.clock.fullName(ctx), "UInt<1>(1)",
          "\"" + printf.format(fmt) + "\"") ++ args
        printfArgs mkString ("printf(", ", ", ")")
      case e: DefInvalid => s"${e.arg.fullName(ctx)} is invalid"
      case e: DefInstance => s"inst ${e.name} of ${e.id.modName}"
      case w: WhenBegin =>
        indent()
        s"when ${w.pred.fullName(ctx)} :"
      case _: WhenEnd =>
        unindent()
        s"skip"
    }
    e.sourceInfo match {
      case SourceLine(filename, line, col) => s"${firrtlLine} @[${filename} ${line}:${col}]"
      case _: NoSourceInfo => firrtlLine
    }
  }

  /** Generates the FIRRTL module declaration.
    */
  private def moduleDecl(m: Component): String = m.id match {
    case _: BlackBox => newline + s"extmodule ${m.name} : "
    case _: Module => newline + s"module ${m.name} : "
  }

  /** Generates the FIRRTL module definition.
    */
  private def moduleDefn(m: Component): String = {
    val body = new StringBuilder
    withIndent {
      for (p <- m.ports)
        body ++= newline + emitPort(p)
      body ++= newline

      m.id match {
        case _: BlackBox =>
          // TODO: BlackBoxes should be empty, but funkiness in Module() means
          // it's not for now. Eventually, this should assert out.
        case _: Module => for (cmd <- m.commands) {
          body ++= newline + emit(cmd, m)
        }
      }
      body ++= newline
    }
    body.toString()
  }

  /** Returns the FIRRTL declaration and body of a module, or nothing if it's a
    * duplicate of something already emitted (on the basis of simple string
    * matching).
    */
  private def emit(m: Component): String = {
    // Generate the body.
    val sb = new StringBuilder
    sb append moduleDecl(m)
    sb append moduleDefn(m)
    sb.result
  }

  private var indentLevel = 0
  private def newline = "\n" + ("  " * indentLevel)
  private def indent(): Unit = indentLevel += 1
  private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
  private def withIndent(f: => Unit) { indent(); f; unindent() }

  private val res = new StringBuilder()
  res ++= s";${Driver.chiselVersionString}\n"
  res ++= s"circuit ${circuit.name} : "
  withIndent { circuit.components.foreach(c => res ++= emit(c)) }
  res ++= newline
}
