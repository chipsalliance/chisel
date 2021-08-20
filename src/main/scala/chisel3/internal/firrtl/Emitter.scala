// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.firrtl

import scala.collection.immutable.LazyList // Needed for 2.12 alias
import firrtl.ir.Serializer

private[chisel3] object Emitter {
<<<<<<< HEAD
  def emit(circuit: Circuit): String = new Emitter(circuit).toString
}

private class Emitter(circuit: Circuit) {
  override def toString: String = res.toString

  private def emitPort(e: Port, topDir: SpecifiedDirection=SpecifiedDirection.Unspecified): String = {
    val resolvedDir = SpecifiedDirection.fromParent(topDir, e.dir)
    val dirString = resolvedDir match {
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Output => "output"
      case SpecifiedDirection.Flip | SpecifiedDirection.Input => "input"
    }
    val clearDir = resolvedDir match {
      case SpecifiedDirection.Input | SpecifiedDirection.Output => true
      case SpecifiedDirection.Unspecified | SpecifiedDirection.Flip => false
    }
    s"$dirString ${e.id.getRef.name} : ${emitType(e.id, clearDir)}"
  }

  private def emitType(d: Data, clearDir: Boolean = false): String = d match {
    case d: Clock => "Clock"
    case _: AsyncReset => "AsyncReset"
    case _: ResetType => "Reset"
    case d: chisel3.core.EnumType => s"UInt${d.width}"
    case d: UInt => s"UInt${d.width}"
    case d: SInt => s"SInt${d.width}"
    case d: FixedPoint => s"Fixed${d.width}${d.binaryPoint}"
    case d: Interval =>
      val binaryPointString = d.binaryPoint match {
        case UnknownBinaryPoint => ""
        case KnownBinaryPoint(value) => s".$value"
      }
      d.toType
    case d: Analog => s"Analog${d.width}"
    case d: Vec[_] => s"${emitType(d.sample_element, clearDir)}[${d.length}]"
    case d: Record => {
      val childClearDir = clearDir ||
          d.specifiedDirection == SpecifiedDirection.Input || d.specifiedDirection == SpecifiedDirection.Output
      def eltPort(elt: Data): String = (childClearDir, firrtlUserDirOf(elt)) match {
        case (true, _) =>
          s"${elt.getRef.name} : ${emitType(elt, true)}"
        case (false, SpecifiedDirection.Unspecified | SpecifiedDirection.Output) =>
          s"${elt.getRef.name} : ${emitType(elt, false)}"
        case (false, SpecifiedDirection.Flip | SpecifiedDirection.Input) =>
          s"flip ${elt.getRef.name} : ${emitType(elt, false)}"
      }
      d.elements.toIndexedSeq.reverse.map(e => eltPort(e._2)).mkString("{", ", ", "}")
    }
  }

  private def firrtlUserDirOf(d: Data): SpecifiedDirection = d match {
    case d: Vec[_] =>
      SpecifiedDirection.fromParent(d.specifiedDirection, firrtlUserDirOf(d.sample_element))
    case d => d.specifiedDirection
  }

  private def emit(e: Command, ctx: Component): String = {
    val firrtlLine = e match {
      case e: DefPrim[_] => s"node ${e.name} = ${e.op.name}(${e.args.map(_.fullName(ctx)).mkString(", ")})"
      case e: DefWire => s"wire ${e.name} : ${emitType(e.id)}"
      case e: DefReg => s"reg ${e.name} : ${emitType(e.id)}, ${e.clock.fullName(ctx)}"
      case e: DefRegInit => s"reg ${e.name} : ${emitType(e.id)}, ${e.clock.fullName(ctx)} with : (reset => (${e.reset.fullName(ctx)}, ${e.init.fullName(ctx)}))"
      case e: DefMemory => s"cmem ${e.name} : ${emitType(e.t)}[${e.size}]"
      case e: DefSeqMemory => s"smem ${e.name} : ${emitType(e.t)}[${e.size}], ${e.readUnderWrite}"
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
      case e: Verification => s"${e.op}(${e.clock.fullName(ctx)}, ${e.predicate.fullName(ctx)}, " +
        s"UInt<1>(1), " + "\"" + s"${printf.format(e.message)}" + "\")"
      case e: DefInvalid => s"${e.arg.fullName(ctx)} is invalid"
      case e: DefInstance => s"inst ${e.name} of ${e.id.name}"
      case w: WhenBegin =>
        // When consequences are always indented
        indent()
        s"when ${w.pred.fullName(ctx)} :"
      case w: WhenEnd =>
        // If a when has no else, the indent level must be reset to the enclosing block
        unindent()
        if (!w.hasAlt) { for (i <- 0 until w.firrtlDepth) { unindent() } }
        s"skip"
      case a: AltBegin =>
        // Else blocks are always indented
        indent()
        s"else :"
      case o: OtherwiseEnd =>
        // Chisel otherwise: ends all FIRRTL associated a Chisel when, resetting indent level
        for (i <- 0 until o.firrtlDepth) { unindent() }
        s"skip"
    }
    firrtlLine + e.sourceInfo.makeMessage(" " + _)
=======
  def emit(circuit: Circuit): String = {
    val fcircuit = Converter.convertLazily(circuit)
<<<<<<< HEAD
    fir.Serializer.serialize(fcircuit)
>>>>>>> 73bd4ee6 (Remove chisel3's own firrtl Emitter, use firrtl Serializer)
=======
    Serializer.serialize(fcircuit)
  }

  def emitLazily(circuit: Circuit): Iterable[String] = {
    val result = LazyList(s"circuit ${circuit.name} :\n")
    val modules = circuit.components.view.map(Converter.convert)
    val moduleStrings = modules.flatMap { m =>
      Array(Serializer.serialize(m, 1), "\n\n")
    }
    result ++ moduleStrings
>>>>>>> d9c30ea0 (Emit .fir lazily, overcomes JVM 2 GiB String limit)
  }
}

