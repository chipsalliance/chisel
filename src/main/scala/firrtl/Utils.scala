// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 *  - Find way to have generic map function instead of mapE and mapS under Stmt implicits
 */

package firrtl

import scala.collection.mutable.StringBuilder
import scala.reflect.runtime.universe._

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
        case v: UIntValue => s"UInt${v.width.serialize}(${v.value.serialize})"
        case v: SIntValue => s"SInt${v.width.serialize}(${v.value.serialize})"
        case r: Ref => r.name
        case s: Subfield => s"${s.exp.serialize}.${s.name}"
        case s: Subindex => s"${s.exp.serialize}[${s.value}]"
        case p: DoPrimOp => 
          s"${p.op.serialize}(" + (p.args.map(_.serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
      } 

    def map(f: Exp => Exp): Exp = 
      exp match {
        case s: Subfield => Subfield(f(s.exp), s.name, s.tpe)
        case s: Subindex => Subindex(f(s.exp), s.value)
        case p: DoPrimOp => DoPrimOp(p.op, p.args.map(f), p.consts)
        case e: Exp => e
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

  // Some Scala implicit magic to solve type erasure on Stmt map function overloading
  private trait StmtMagnet {
    def map(stmt: Stmt): Stmt
  }
  private object StmtMagnet {
    implicit def forStmt(f: Stmt => Stmt) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt =
        stmt match {
          case w: When => When(w.info, w.pred, f(w.conseq), f(w.alt))
          case b: Block => Block(b.stmts.map(f))
          case s: Stmt => s
        }
    }
    implicit def forExp(f: Exp => Exp) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt =
        stmt match {
          case r: DefReg => DefReg(r.info, r.name, r.tpe, f(r.clock), f(r.reset))
          case m: DefMemory => DefMemory(m.info, m.name, m.seq, m.tpe, f(m.clock))
          case i: DefInst => DefInst(i.info, i.name, f(i.module))
          case n: DefNode => DefNode(n.info, n.name, f(n.value))
          case a: DefAccessor => DefAccessor(a.info, a.name, a.dir, f(a.source), f(a.index))
          case o: OnReset => OnReset(o.info, f(o.lhs), f(o.rhs))
          case c: Connect => Connect(c.info, f(c.lhs), f(c.rhs))
          case b: BulkConnect => BulkConnect(b.info, f(b.lhs), f(b.rhs))
          case w: When => When(w.info, f(w.pred), w.conseq, w.alt)
          case a: Assert => Assert(a.info, f(a.pred))
          case s: Stmt => s 
        }
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

    // Using implicit types to allow overloading of function type to map, see StmtMagnet above
    def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Stmt = magnet(f).map(stmt)
    
  }

  implicit class WidthUtils(w: Width) {
    def serialize(): String = 
      w match {
        case UnknownWidth => ""
        case w: IntWidth => s"<${w.width.toString}>"
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
          case t: UIntType => s"UInt${t.width.serialize}"
          case t: SIntType => s"SInt${t.width.serialize}"
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
