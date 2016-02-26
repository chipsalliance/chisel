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

import firrtl.PrimOps._
import firrtl.Utils._

object Serialize {

  implicit class BigIntSerialize(bi: BigInt){
    def serialize: String =
      if (bi < BigInt(0)) "\"h" + bi.toString(16).substring(1) + "\""
      else "\"h" + bi.toString(16) + "\""
  }
   
  implicit class PrimOpSerialize(op: PrimOp) {
    def serialize: String = op.getString
  }

  implicit class ExpSerialize(exp: Expression) {
    def serialize: String = {
      exp match {
        case v: UIntValue => s"UInt${v.width.serialize}(${v.value.serialize})"
        case v: SIntValue => s"SInt${v.width.serialize}(${v.value.serialize})"
        case r: Ref => r.name
        case s: SubField => s"${s.exp.serialize}.${s.name}"
        case s: SubIndex => s"${s.exp.serialize}[${s.value}]"
        case s: SubAccess => s"${s.exp.serialize}[${s.index.serialize}]"
        case m: Mux => s"mux(${m.cond.serialize}, ${m.tval.serialize}, ${m.fval.serialize})"
        case v: ValidIf => s"validif(${v.cond.serialize}, ${v.value.serialize})"
        case p: DoPrim => 
          s"${p.op.serialize}(" + (p.args.map(_.serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
        case r: WRef => r.name
        case s: WSubField => s"${s.exp.serialize}.${s.name}"
        case s: WSubIndex => s"${s.exp.serialize}[${s.value}]"
        case s: WSubAccess => s"${s.exp.serialize}[${s.index.serialize}]"
        case r: WVoid => "VOID"
      } 
    }
  }

  implicit class StmtSerialize(stmt: Stmt) {
    def serialize: String = {
      stmt match {
        case w: DefWire => s"wire ${w.name} : ${w.tpe.serialize}"
        case r: DefRegister => 
          val str = new StringBuilder(s"reg ${r.name} : ${r.tpe.serialize}, ${r.clock.serialize} with : ")
          withIndent {
            str ++= newline + s"reset => (${r.reset.serialize}, ${r.init.serialize})"
          }
          str.toString
        case i: DefInstance => s"inst ${i.name} of ${i.module}"
        case i: WDefInstance => s"inst ${i.name} of ${i.module}"
        case m: DefMemory => {
          val str = new StringBuilder(s"mem ${m.name} : ")
          withIndent {
            str ++= newline + 
              s"data-type => ${m.data_type.serialize}" + newline +
              s"depth => ${m.depth}" + newline +
              s"read-latency => ${m.read_latency}" + newline +
              s"write-latency => ${m.write_latency}" + newline +
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
        case n: DefNode => s"node ${n.name} = ${n.value.serialize}"
        case c: Connect => s"${c.loc.serialize} <= ${c.exp.serialize}"
        case b: BulkConnect => s"${b.loc.serialize} <- ${b.exp.serialize}"
        case w: Conditionally => {
          var str = new StringBuilder(s"when ${w.pred.serialize} : ")
          withIndent { str ++= newline + w.conseq.serialize }
          w.alt match {
             case s:Empty => str.result
             case s => {
               str ++= newline + "else :"
               withIndent { str ++= newline + w.alt.serialize }
               str.result
               }
          }
        }
        case b: Begin => {
          val s = new StringBuilder
          for (i <- 0 until b.stmts.size) {
            if (i != 0) s ++= newline ++ b.stmts(i).serialize
            else s ++= b.stmts(i).serialize
          }
          s.result
        } 
        case i: IsInvalid => s"${i.exp.serialize} is invalid"
        case s: Stop => s"stop(${s.clk.serialize}, ${s.en.serialize}, ${s.ret})"
        case p: Print => {
          val q = '"'.toString
          s"printf(${p.clk.serialize}, ${p.en.serialize}, ${q}${p.string}${q}" + 
                        (if (p.args.nonEmpty) p.args.map(_.serialize).mkString(", ", ", ", "") else "") + ")"
        }
        case s:Empty => "skip"
        case s:CDefMemory => {
          if (s.seq) s"smem ${s.name} : ${s.tpe} [${s.size}]"
          else s"cmem ${s.name} : ${s.tpe} [${s.size}]"
        }
        case s:CDefMPort => {
          val dir = s.direction match {
             case MInfer => "infer"
             case MRead => "read"
             case MWrite => "write"
             case MReadWrite => "rdwr"
          }
          s"${dir} mport ${s.name} = ${s.mem}[${s.exps(0)}], s.exps(1)"
        }
      } 
    }
  }

  implicit class WidthSerialize(w: Width) {
    def serialize: String = {
      w match {
        case w:UnknownWidth => "" 
        case w: IntWidth => s"<${w.width.toString}>"
        case w: VarWidth => s"<${w.name}>"
      } 
    }
  }

  implicit class FlipSerialize(f: Flip) {
    def serialize: String = {
      f match {
        case REVERSE => "flip "
        case DEFAULT => ""
      } 
    }
  }

  implicit class FieldSerialize(field: Field) {
    def serialize: String =
      s"${field.flip.serialize}${field.name} : ${field.tpe.serialize}"
  }

  implicit class TypeSerialize(t: Type) {
    def serialize: String = {
      val commas = ", " // for mkString in BundleType
      t match {
        case c:ClockType => "Clock"
        case u:UnknownType => "?"
        case t: UIntType => s"UInt${t.width.serialize}"
        case t: SIntType => s"SInt${t.width.serialize}"
        case t: BundleType => s"{ ${t.fields.map(_.serialize).mkString(commas)}}"
        case t: VectorType => s"${t.tpe.serialize}[${t.size}]"
      } 
    }
  }

  implicit class DirectionSerialize(d: Direction) {
    def serialize: String = {
      d match {
        case INPUT => "input"
        case OUTPUT => "output"
      } 
    }
  }

  implicit class PortSerialize(p: Port) {
    def serialize: String =
      s"${p.direction.serialize} ${p.name} : ${p.tpe.serialize}"
  }

  implicit class ModuleSerialize(m: Module) {
    def serialize: String = {
      m match {
         case m:InModule => {
            var s = new StringBuilder(s"module ${m.name} : ")
            withIndent {
              s ++= m.ports.map(newline ++ _.serialize).mkString
              s ++= newline ++ m.body.serialize
            }
            s.toString
         }
         case m:ExModule => {
            var s = new StringBuilder(s"extmodule ${m.name} : ")
            withIndent {
              s ++= m.ports.map(newline ++ _.serialize).mkString
              s ++= newline
            }
            s.toString
         }
      }
    }
  }

  implicit class CircuitSerialize(c: Circuit) {
    def serialize: String = {
      var s = new StringBuilder(s"circuit ${c.main} : ")
      withIndent { s ++= newline ++ c.modules.map(_.serialize).mkString(newline + newline) }
      s ++= newline ++ newline
      s.toString
    }
  }

  private var indentLevel = 0
  private def newline = "\n" + ("  " * indentLevel)
  private def indent(): Unit = indentLevel += 1
  private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
  private def withIndent(f: => Unit) { indent(); f; unindent() }
}
