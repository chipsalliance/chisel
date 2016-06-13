/*
Copyright (c) 2014 - 2016 The Regents of the University of
California (Regents). All Rights Reserved.  Redistribution and use in
source and binary forms, with or without modification, are permitted
provided that the following conditions are met:
   * Redistributions of source code must retain the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer.
   * Redistributions in binary form must reproduce the above
     copyright notice, this list of conditions and the following
     two paragraphs of disclaimer in the documentation and/or other materials
     provided with the distribution.
   * Neither the name of the Regents nor the names of its contributors
     may be used to endorse or promote products derived from this
     software without specific prior written permission.
IN NO EVENT SHALL REGENTS BE LIABLE TO ANY PARTY FOR DIRECT, INDIRECT,
SPECIAL, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, INCLUDING LOST PROFITS,
ARISING OUT OF THE USE OF THIS SOFTWARE AND ITS DOCUMENTATION, EVEN IF
REGENTS HAS BEEN ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
REGENTS SPECIFICALLY DISCLAIMS ANY WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE. THE SOFTWARE AND ACCOMPANYING DOCUMENTATION, IF
ANY, PROVIDED HEREUNDER IS PROVIDED "AS IS". REGENTS HAS NO OBLIGATION
TO PROVIDE MAINTENANCE, SUPPORT, UPDATES, ENHANCEMENTS, OR
MODIFICATIONS.
*/

package firrtl

import firrtl.ir._
import firrtl.PrimOps._
import firrtl.Utils._

private object Serialize {
  def serialize(root: FirrtlNode): String = {
    lazy val ser = new Serialize
    root match {
      case r: PrimOp => ser.serialize(r)
      case r: Expression => ser.serialize(r)
      case r: Statement => ser.serialize(r)
      case r: Width => ser.serialize(r)
      case r: Orientation => ser.serialize(r)
      case r: Field => ser.serialize(r)
      case r: Type => ser.serialize(r)
      case r: Direction => ser.serialize(r)
      case r: Port => ser.serialize(r)
      case r: DefModule => ser.serialize(r)
      case r: Circuit => ser.serialize(r)
      case r: StringLit => ser.serialize(r)
      case _ => throw new Exception("serialize called on unknown FirrtlNode!")
    }
  }
  /** Creates new instance of Serialize */
  def apply() = new Serialize
}

class Serialize {
  def serialize(bi: BigInt): String =
    if (bi < BigInt(0)) "\"h" + bi.toString(16).substring(1) + "\""
    else "\"h" + bi.toString(16) + "\""

  def serialize(info: Info): String = " " + info.toString

  def serialize(op: PrimOp): String = op.toString

  def serialize(lit: StringLit): String = FIRRTLStringLitHandler.escape(lit)

  def serialize(exp: Expression): String = {
    exp match {
      case v: UIntLiteral => s"UInt${serialize(v.width)}(${serialize(v.value)})"
      case v: SIntLiteral => s"SInt${serialize(v.width)}(${serialize(v.value)})"
      case r: Reference => r.name
      case s: SubField => s"${serialize(s.expr)}.${s.name}"
      case s: SubIndex => s"${serialize(s.expr)}[${s.value}]"
      case s: SubAccess => s"${serialize(s.expr)}[${serialize(s.index)}]"
      case m: Mux => s"mux(${serialize(m.cond)}, ${serialize(m.tval)}, ${serialize(m.fval)})"
      case v: ValidIf => s"validif(${serialize(v.cond)}, ${serialize(v.value)})"
      case p: DoPrim =>
        s"${serialize(p.op)}(" + (p.args.map(serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
      case r: WRef => r.name
      case s: WSubField => s"${serialize(s.exp)}.${s.name}"
      case s: WSubIndex => s"${serialize(s.exp)}[${s.value}]"
      case s: WSubAccess => s"${serialize(s.exp)}[${serialize(s.index)}]"
      case r: WVoid => "VOID"
    }
  }

  def serialize(stmt: Statement): String = {
    stmt match {
      case w: DefWire => s"wire ${w.name} : ${serialize(w.tpe)}${w.info}"
      case r: DefRegister =>
        val str = new StringBuilder(s"reg ${r.name} : ${serialize(r.tpe)}, ${serialize(r.clock)} with :")
        withIndent {
          str ++= newline + s"reset => (${serialize(r.reset)}, ${serialize(r.init)})${r.info}"
        }
        str.toString
      case i: DefInstance => s"inst ${i.name} of ${i.module}${i.info}"
      case i: WDefInstance => s"inst ${i.name} of ${i.module}${i.info}"
      case m: DefMemory => {
        val str = new StringBuilder(s"mem ${m.name} :${m.info}")
        withIndent {
          str ++= newline +
            s"data-type => ${serialize(m.dataType)}" + newline +
            s"depth => ${m.depth}" + newline +
            s"read-latency => ${m.readLatency}" + newline +
            s"write-latency => ${m.writeLatency}" + newline +
            (if (m.readers.nonEmpty) m.readers.map(r => s"reader => ${r}").mkString(newline) + newline
             else "") +
            (if (m.writers.nonEmpty) m.writers.map(w => s"writer => ${w}").mkString(newline) + newline
             else "") +
            (if (m.readwriters.nonEmpty) m.readwriters.map(rw => s"readwriter => ${rw}").mkString(newline) + newline
             else "") +
            s"read-under-write => undefined"
        }
        str.result
      }
      case n: DefNode => s"node ${n.name} = ${serialize(n.value)}${n.info}"
      case c: Connect => s"${serialize(c.loc)} <= ${serialize(c.expr)}${c.info}"
      case p: PartialConnect => s"${serialize(p.loc)} <- ${serialize(p.expr)}${p.info}"
      case w: Conditionally => {
        var str = new StringBuilder(s"when ${serialize(w.pred)} :${w.info}")
        withIndent { str ++= newline + serialize(w.conseq) }
        w.alt match {
          case EmptyStmt => str.result
          case s => {
            str ++= newline + "else :"
            withIndent { str ++= newline + serialize(w.alt) }
            str.result
            }
        }
      }
      case b: Begin => {
        val s = new StringBuilder
        for (i <- 0 until b.stmts.size) {
          if (i != 0) s ++= newline ++ serialize(b.stmts(i))
          else s ++= serialize(b.stmts(i))
        }
        s.result
      }
      case i: IsInvalid => s"${serialize(i.expr)} is invalid${i.info}"
      case s: Stop => s"stop(${serialize(s.clk)}, ${serialize(s.en)}, ${s.ret})${s.info}"
      case p: Print => {
        val q = '"'.toString
        s"printf(${serialize(p.clk)}, ${serialize(p.en)}, ${q}${serialize(p.string)}${q}" +
                      (if (p.args.nonEmpty) p.args.map(serialize).mkString(", ", ", ", "") else "") +
                      s")${p.info}"
      }
      case EmptyStmt => "skip"
      case s: CDefMemory => {
        if (s.seq) s"smem ${s.name} : ${serialize(s.tpe)} [${s.size}]${s.info}"
        else s"cmem ${s.name} : ${serialize(s.tpe)} [${s.size}]${s.info}"
      }
      case s: CDefMPort => {
        val dir = s.direction match {
          case MInfer => "infer"
          case MRead => "read"
          case MWrite => "write"
          case MReadWrite => "rdwr"
        }
        s"${dir} mport ${s.name} = ${s.mem}[${serialize(s.exps(0))}], ${serialize(s.exps(1))}${s.info}"
      }
    }
  }

  def serialize(w: Width): String = {
    w match {
      case UnknownWidth => ""
      case w: IntWidth => s"<${w.width.toString}>"
      case w: VarWidth => s"<${w.name}>"
    }
  }

  def serialize(f: Orientation): String = {
    f match {
      case Flip => "flip "
      case Default => ""
    }
  }

  def serialize(field: Field): String =
    s"${serialize(field.flip)}${field.name} : ${serialize(field.tpe)}"

  def serialize(t: Type): String = {
    val commas = ", " // for mkString in BundleType
    t match {
      case ClockType => "Clock"
      case UnknownType => "?"
      case t: UIntType => s"UInt${serialize(t.width)}"
      case t: SIntType => s"SInt${serialize(t.width)}"
      case t: BundleType => s"{ ${t.fields.map(serialize).mkString(commas)}}"
      case t: VectorType => s"${serialize(t.tpe)}[${t.size}]"
    }
  }

  def serialize(d: Direction): String = {
    d match {
      case Input => "input"
      case Output => "output"
    }
  }

  def serialize(p: Port): String =
    s"${serialize(p.direction)} ${p.name} : ${serialize(p.tpe)}${p.info}"

  def serialize(m: DefModule): String = {
    m match {
      case m: Module => {
        var s = new StringBuilder(s"module ${m.name} :${m.info}")
        withIndent {
          s ++= m.ports.map(newline ++ serialize(_)).mkString
          s ++= newline ++ serialize(m.body)
        }
        s.toString
      }
      case m: ExtModule => {
        var s = new StringBuilder(s"extmodule ${m.name} :${m.info}")
        withIndent {
          s ++= m.ports.map(newline ++ serialize(_)).mkString
          s ++= newline
        }
        s.toString
      }
    }
  }

  def serialize(c: Circuit): String = {
    var s = new StringBuilder(s"circuit ${c.main} :${c.info}")
    withIndent { s ++= newline ++ c.modules.map(serialize).mkString(newline + newline) }
    s ++= newline ++ newline
    s.toString
  }

  private var indentLevel = 0
  private def newline = "\n" + ("  " * indentLevel)
  private def indent(): Unit = indentLevel += 1
  private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
  private def withIndent(f: => Unit) { indent(); f; unindent() }
}
