// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 */

package firrtl

import scala.collection.mutable.StringBuilder

object Utils {
  
  implicit class BigIntUtils(bi: BigInt){
    def serialize(): String = 
      "\"h0" + bi.toString(16) + "\""
  }

  implicit class PrimOpUtils(op: PrimOp) {
    def serialize(): String = {
      op match { 
        case Add => "add"
        case Sub => "sub"
        case Addw => "addw"
        case Subw => "subw"
        case Mul => "mul"
        case Div => "div"
        case Mod => "mod"
        case Quo => "quo"
        case Rem => "rem"
        case Lt => "lt"
        case Leq => "leq"
        case Gt => "gt"
        case Geq => "geq"
        case Eq => "eq"
        case Neq => "neq"
        case Mux => "mux"
        case Pad => "pad"
        case AsUInt => "asUInt"
        case AsSInt => "asSInt"
        case Shl => "shl"
        case Shr => "shr"
        case Dshl => "dshl"
        case Dshr => "dshr"
        case Cvt => "cvt"
        case Neg => "neg"
        case Not => "not"
        case And => "and"
        case Or => "or"
        case Xor => "xor"
        case Andr => "andr"
        case Orr => "orr"
        case Xorr => "xorr"
        case Cat => "cat"
        case Bit => "bit"
        case Bits => "bits"
      } 
    }
  }

  implicit class ExpUtils(exp: Exp) {
    def serialize(): String = 
      exp match {
        case v: UIntValue => s"UInt<${v.width}>(${v.value.serialize})"
        case v: SIntValue => s"SInt<${v.width}>(${v.value.serialize})"
        case r: Ref => r.name
        case s: Subfield => s"${s.exp.serialize}.${s.name}"
        case s: Subindex => s"${s.exp.serialize}[${s.value}]"
        case p: DoPrimOp => 
          s"${p.op.serialize}(" + (p.args.map(_.serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
      } 
  }
  
  // AccessorDir
  implicit class AccessorDirUtils(dir: AccessorDir) {
    def serialize(): String = 
      dir match {
        case Infer => "infer"
        case Read => "read"
        case Write => "write"
        case RdWr => "rdwr"
      } 
  }


  implicit class StmtUtils(stmt: Stmt) {
    def serialize(): String = 
      stmt match {
        case w: DefWire => s"wire ${w.name} : ${w.tpe.serialize}"
        case r: DefReg => s"reg ${r.name} : ${r.tpe.serialize}, ${r.clock.serialize}, ${r.reset.serialize}"
        case m: DefMemory => (if(m.seq) "smem" else "cmem") + 
          s" ${m.name} : ${m.tpe.serialize}, ${m.clock.serialize}"
        case i: DefInst => s"inst ${i.name} of ${i.module.serialize}"
        case n: DefNode => s"node ${n.name} = ${n.value.serialize}"
        case p: DefPoison => s"poison ${p.name} : ${p.tpe.serialize}"
        case a: DefAccessor => s"${a.dir.serialize} accessor ${a.name} = ${a.source.serialize}[${a.index.serialize}]"
        case c: Connect => s"${c.lhs.serialize} := ${c.rhs.serialize}"
        case o: OnReset => s"onreset ${o.lhs.serialize} := ${o.rhs.serialize}"
        case b: BulkConnect => s"${b.lhs.serialize} <> ${b.rhs.serialize}"
        case w: When => {
          var str = new StringBuilder(s"when ${w.pred.serialize} : ")
          withIndent { str ++= w.conseq.serialize }
          if( w.alt != EmptyStmt ) {
            str ++= newline + "else :"
            withIndent { str ++= w.alt.serialize }
          }
          str.result
        }
        //case b: Block => b.stmts.map(newline ++ _.serialize).mkString
        case b: Block => {
          val s = new StringBuilder
          b.stmts.foreach { s ++= newline ++ _.serialize }
          s.result
        }
        case a: Assert => s"assert ${a.pred.serialize}"
        case EmptyStmt => "skip"
      } 
  }

  implicit class WidthUtils(w: Width) {
    def serialize(): String = 
      w match {
        case UnknownWidth => "?"
        case w: IntWidth => w.width.toString
      } 
  }

  implicit class FieldDirUtils(dir: FieldDir) {
    def serialize(): String = 
      dir match {
        case Reverse => "flip "
        case Default => ""
      } 
  }

  implicit class FieldUtils(field: Field) {
    def serialize(): String = 
      s"${field.dir.serialize} ${field.name} : ${field.tpe.serialize}"
  }

  implicit class TypeUtils(t: Type) {
    def serialize(): String = {
      val commas = ", " // for mkString in BundleType
        t match {
          case ClockType => "Clock"
          case UnknownType => "UnknownType"
          case t: UIntType => s"UInt<${t.width.serialize}>"
          case t: SIntType => s"SInt<${t.width.serialize}>"
          case t: BundleType => s"{ ${t.fields.map(_.serialize).mkString(commas)} }"
          case t: VectorType => s"${t.tpe.serialize}[${t.size}]"
        } 
    }
  }

  implicit class PortDirUtils(p: PortDir) {
    def serialize(): String = 
      p match {
        case Input => "input"
        case Output => "output"
      } 
  }

  implicit class PortUtils(p: Port) {
    def serialize(): String = 
      s"${p.dir.serialize} ${p.name} : ${p.tpe.serialize}"
  }

  implicit class ModuleUtils(m: Module) {
    def serialize(): String = {
      var s = new StringBuilder(s"module ${m.name} : ")
      withIndent {
        s ++= m.ports.map(newline ++ _.serialize).mkString
        s ++= newline ++ m.stmt.serialize
      }
      s.toString
    }
  }

  implicit class CircuitUtils(c: Circuit) {
    def serialize(): String = {
      var s = new StringBuilder(s"circuit ${c.name} : ")
      //withIndent { c.modules.foreach(s ++= newline ++ newline ++ _.serialize) }
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
