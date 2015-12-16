// See LICENSE for license details.

package Chisel.firrtl
import Chisel._

/** Class which emits the internal circuit data structures as a string.
  */
private class Emitter(circuit: Circuit) {
  override def toString: String = res.toString

  /** Returns the FIRRTL representation of a port.
    */
  private def emitPort(e: Port): String =
    s"${e.dir} ${e.id.getRef.name} : ${e.id.toType}"

  /** Returns the FIRRTL representation of a statement / declaration.
    */
  private def emit(e: Command, ctx: Component): String = e match {
    case e: DefPrim[_] => s"node ${e.name} = ${e.op.name}(${e.args.map(_.fullName(ctx)).reduce(_ + ", " + _)})"
    case e: DefWire => s"wire ${e.name} : ${e.id.toType}"
    case e: DefPoison[_] => s"poison ${e.name} : ${e.id.toType}"
    case e: DefRegister => s"reg ${e.name} : ${e.id.toType}, ${e.clock.fullName(ctx)}, ${e.reset.fullName(ctx)}"
    case e: DefMemory => s"cmem ${e.name} : ${e.t.toType}[${e.size}], ${e.clock.fullName(ctx)}"
    case e: DefSeqMemory => s"smem ${e.name} : ${e.t.toType}[${e.size}], ${e.clock.fullName(ctx)}"
    case e: DefAccessor[_] => s"infer accessor ${e.name} = ${e.source.fullName(ctx)}[${e.index.fullName(ctx)}]"
    case e: Connect => s"${e.loc.fullName(ctx)} := ${e.exp.fullName(ctx)}"
    case e: BulkConnect => s"${e.loc1.fullName(ctx)} <> ${e.loc2.fullName(ctx)}"
    case e: ConnectInit => s"onreset ${e.loc.fullName(ctx)} := ${e.exp.fullName(ctx)}"
    case e: Stop => s"stop(${e.clk.fullName(ctx)}, ${e.ret})"
    case e: Printf => s"""printf(${e.clk.fullName(ctx)}, "${e.format}"${e.ids.map(_.fullName(ctx)).fold(""){_ + ", " + _}})"""
    case e: DefInstance => {
      val modName = moduleMap.get(e.id.name).get
      s"inst ${e.name} of $modName"
    }

    case w: WhenBegin =>
      indent()
      s"when ${w.pred.fullName(ctx)} :"
    case _: WhenElse =>
      indent()
      "else :"
    case _: WhenEnd =>
      unindent()
      "skip"
  }

  // Map of FIRRTL Module body to FIRRTL id, if it has been emitted already.
  private val bodyMap = collection.mutable.HashMap[StringBuilder, String]()
  // Map of Component name to FIRRTL id.
  private val moduleMap = collection.mutable.HashMap[String, String]()

  /** Returns the FIRRTL declaration and body of a module, or nothing if it's a
    * duplicate of something already emitted (on the basis of simple string
    * matching).
    */
  private def emit(m: Component): String = {
    // Generate the body.
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

    bodyMap get body match {
      case Some(name) =>
        moduleMap(m.name) = name
        ""
      case None =>
        require(!(moduleMap contains m.name),
            "emitting module with same name but different contents")
        moduleMap(m.name) = m.name
        bodyMap(body) = m.name

        val decl: String = m.id match {
          case _: BlackBox => s"extmodule ${m.name} : "
          case _: Module => s"module ${m.name} : "
        }
        newline + decl + body.toString()
    }
  }

  private var indentLevel = 0
  private def newline = "\n" + ("  " * indentLevel)
  private def indent(): Unit = indentLevel += 1
  private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
  private def withIndent(f: => Unit) { indent(); f; unindent() }

  private val res = new StringBuilder(s"circuit ${circuit.name} : ")
  withIndent { circuit.components.foreach(c => res ++= emit(c)) }
  res ++= newline
}
