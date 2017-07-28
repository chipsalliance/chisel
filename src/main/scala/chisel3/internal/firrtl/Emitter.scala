// See LICENSE for license details.

package chisel3.internal.firrtl
import chisel3._
import chisel3.core.UserDirection
import chisel3.experimental._
import chisel3.internal.sourceinfo.{NoSourceInfo, SourceLine}

private[chisel3] object Emitter {
  def emit(circuit: Circuit): String = new Emitter(circuit).toString
}

private class Emitter(circuit: Circuit) {
  override def toString: String = res.toString

  private def emitPort(e: Port, topDir: UserDirection=UserDirection.Unspecified): String = {
    val resolvedDir = UserDirection.fromParent(topDir, e.dir)
    val dirString = resolvedDir match {
      case UserDirection.Unspecified | UserDirection.Output => "output"
      case UserDirection.Flip | UserDirection.Input => "input"
    }
    val clearDir = resolvedDir match {
      case UserDirection.Input | UserDirection.Output => true
      case UserDirection.Unspecified | UserDirection.Flip => false
    }
    s"$dirString ${e.id.getRef.name} : ${emitType(e.id, clearDir)}"
  }

  private def emitType(d: Data, clearDir: Boolean = false): String = d match {
    case d: Clock => "Clock"
    case d: UInt => s"UInt${d.width}"
    case d: SInt => s"SInt${d.width}"
    case d: FixedPoint => s"Fixed${d.width}${d.binaryPoint}"
    case d: Analog => s"Analog${d.width}"
    case d: Vec[_] => s"${emitType(d.sample_element, clearDir)}[${d.length}]"
    case d: Record => {
      val childClearDir = clearDir ||
          d.userDirection == UserDirection.Input || d.userDirection == UserDirection.Output
      def eltPort(elt: Data): String = (childClearDir, firrtlUserDirOf(elt)) match {
        case (true, _) =>
          s"${elt.getRef.name} : ${emitType(elt, true)}"
        case (false, UserDirection.Unspecified | UserDirection.Output) =>
          s"${elt.getRef.name} : ${emitType(elt, false)}"
        case (false, UserDirection.Flip | UserDirection.Input) =>
          s"flip ${elt.getRef.name} : ${emitType(elt, false)}"
      }
      d.elements.toIndexedSeq.reverse.map(e => eltPort(e._2)).mkString("{", ", ", "}")
    }
  }

  private def firrtlUserDirOf(d: Data): UserDirection = d match {
    case d: Vec[_] =>
      UserDirection.fromParent(d.userDirection, firrtlUserDirOf(d.sample_element))
    case d => d.userDirection
  }

  private def emit(e: Command, ctx: Component): String = {
    val firrtlLine = e match {
      case e: DefPrim[_] => s"node ${e.name} = ${e.op.name}(${e.args.map(_.fullName(ctx)).mkString(", ")})"
      case e: DefWire => s"wire ${e.name} : ${emitType(e.id)}"
      case e: DefReg => s"reg ${e.name} : ${emitType(e.id)}, ${e.clock.fullName(ctx)}"
      case e: DefRegInit => s"reg ${e.name} : ${emitType(e.id)}, ${e.clock.fullName(ctx)} with : (reset => (${e.reset.fullName(ctx)}, ${e.init.fullName(ctx)}))"
      case e: DefMemory => s"cmem ${e.name} : ${emitType(e.t)}[${e.size}]"
      case e: DefSeqMemory => s"smem ${e.name} : ${emitType(e.t)}[${e.size}]"
      case e: DefMemPort[_] => s"${e.dir} mport ${e.name} = ${e.source.fullName(ctx)}[${e.index.fullName(ctx)}], ${e.clock.fullName(ctx)}"
      case e: Connect => s"${e.loc.fullName(ctx)} <= ${e.exp.fullName(ctx)}"
      case e: BulkConnect => s"${e.loc1.fullName(ctx)} <- ${e.loc2.fullName(ctx)}"
      case e: Attach => e.locs.map(_.fullName(ctx)).mkString("attach (", ", ", ")")
      case e: Stop => s"stop(${e.clock.fullName(ctx)}, UInt<1>(1), ${e.ret})"
      case e: Printf =>
        val (fmt, args) = e.pable.unpack(ctx)
        val printfArgs = Seq(e.clock.fullName(ctx), "UInt<1>(1)",
          "\"" + printf.format(fmt) + "\"") ++ args
        printfArgs mkString ("printf(", ", ", ")")
      case e: DefInvalid => s"${e.arg.fullName(ctx)} is invalid"
      case e: DefInstance => s"inst ${e.name} of ${e.id.name}"
      case w: WhenBegin =>
        indent()
        s"when ${w.pred.fullName(ctx)} :"
      case _: WhenEnd =>
        unindent()
        s"skip"
    }
    firrtlLine + e.sourceInfo.makeMessage(" " + _)
  }

  private def emitParam(name: String, p: Param): String = {
    val str = p match {
      case IntParam(value) => value.toString
      case DoubleParam(value) => value.toString
      case StringParam(str) => "\"" + str + "\""
      case RawParam(str) => "'" + str + "'"
    }
    s"parameter $name = $str"
  }

  /** Generates the FIRRTL module declaration.
    */
  private def moduleDecl(m: Component): String = m.id match {
    case _: chisel3.core.BaseBlackBox => newline + s"extmodule ${m.name} : "
    case _: chisel3.core.UserModule => newline + s"module ${m.name} : "
  }

  /** Generates the FIRRTL module definition.
    */
  private def moduleDefn(m: Component): String = {
    val body = new StringBuilder
    withIndent {
      for (p <- m.ports) {
        val portDef = m match {
          case bb: DefBlackBox => emitPort(p, bb.topDir)
          case mod: DefModule => emitPort(p)
        }
        body ++= newline + portDef
      }
      body ++= newline

      m match {
        case bb: DefBlackBox =>
          // Firrtl extmodule can overrule name
          body ++= newline + s"defname = ${bb.id.desiredName}"
          body ++= newline + (bb.params map { case (n, p) => emitParam(n, p) } mkString newline)
        case mod: DefModule => for (cmd <- mod.commands) {
          body ++= newline + emit(cmd, mod)
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
    sb.append(moduleDecl(m))
    sb.append(moduleDefn(m))
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
