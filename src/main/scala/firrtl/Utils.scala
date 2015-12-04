// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 *  - Find way to have generic map function instead of mapE and mapS under Stmt implicits
 */

package firrtl

import scala.collection.mutable.StringBuilder
import java.io.PrintWriter
import Primops._
//import scala.reflect.runtime.universe._

object Utils {

  // Is there a more elegant way to do this?
  private type FlagMap = Map[Symbol, Boolean]
  private val FlagMap = Map[Symbol, Boolean]().withDefaultValue(false)

  def debug(node: AST)(implicit flags: FlagMap): String = {
    if (!flags.isEmpty) {
      var str = ""
      if (flags('types)) {
        val tpe = node.getType
        if( tpe != UnknownType ) str += s"@<t:${tpe.wipeWidth.serialize}>"
      }
      str
    }
    else {
      ""
    }
  }

  implicit class BigIntUtils(bi: BigInt){
    def serialize(implicit flags: FlagMap = FlagMap): String = 
      "\"h" + bi.toString(16) + "\""
  }

  implicit class ASTUtils(ast: AST) {
    def getType(): Type = 
      ast match {
        case e: Exp => e.getType
        case s: Stmt => s.getType
        //case f: Field => f.getType
        case t: Type => t.getType
        case p: Port => p.getType
        case _ => UnknownType
      }
  }

  implicit class PrimopUtils(op: Primop) {
    def serialize(implicit flags: FlagMap = FlagMap): String = op.getString
  }

  implicit class ExpUtils(exp: Exp) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val ret = exp match {
        case v: UIntValue => s"UInt${v.width.serialize}(${v.value.serialize})"
        case v: SIntValue => s"SInt${v.width.serialize}(${v.value.serialize})"
        case r: Ref => r.name
        case s: Subfield => s"${s.exp.serialize}.${s.name}"
        case s: Index => s"${s.exp.serialize}[${s.value}]"
        case p: DoPrimop => 
          s"${p.op.serialize}(" + (p.args.map(_.serialize) ++ p.consts.map(_.toString)).mkString(", ") + ")"
      } 
      ret + debug(exp)
    }

    def map(f: Exp => Exp): Exp = 
      exp match {
        case s: Subfield => Subfield(f(s.exp), s.name, s.tpe)
        case i: Index => Index(f(i.exp), i.value, i.tpe)
        case p: DoPrimop => DoPrimop(p.op, p.args.map(f), p.consts, p.tpe)
        case e: Exp => e
      }

    def getType(): Type = {
      exp match {
        case v: UIntValue => UIntType(UnknownWidth)
        case v: SIntValue => SIntType(UnknownWidth)
        case r: Ref => r.tpe
        case s: Subfield => s.tpe
        case i: Index => i.tpe
        case p: DoPrimop => p.tpe
      }
    }
  }
  
  // AccessorDir
  implicit class AccessorDirUtils(dir: AccessorDir) {
    def serialize(implicit flags: FlagMap = FlagMap): String = 
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
    def serialize(implicit flags: FlagMap = FlagMap): String =
    {
      var ret = stmt match {
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
          s.result + debug(b)
        } 
        case a: Assert => s"assert ${a.pred.serialize}"
        case EmptyStmt => "skip"
      } 
      ret + debug(stmt)
    }

    // Using implicit types to allow overloading of function type to map, see StmtMagnet above
    def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Stmt = magnet(f).map(stmt)
    
    def getType(): Type =
      stmt match {
        case s: DefWire   => s.tpe
        case s: DefReg    => s.tpe
        case s: DefMemory => s.tpe
        case s: DefPoison => s.tpe
        case _ => UnknownType
      }
  }

  implicit class WidthUtils(w: Width) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val s = w match {
        case UnknownWidth => "" //"?"
        case w: IntWidth => s"<${w.width.toString}>"
      } 
      s + debug(w)
    }
  }

  implicit class FieldDirUtils(dir: FieldDir) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val s = dir match {
        case Reverse => "flip"
        case Default => ""
      } 
      s + debug(dir)
    }
    def toPortDir(): PortDir = {
      dir match {
        case Default => Output
        case Reverse => Input
      }
    }
  }

  implicit class FieldUtils(field: Field) {
    def serialize(implicit flags: FlagMap = FlagMap): String = 
      s"${field.dir.serialize} ${field.name} : ${field.tpe.serialize}" + debug(field)

    def getType(): Type = field.tpe
    def toPort(): Port = Port(NoInfo, field.name, field.dir.toPortDir, field.tpe)
  }

  implicit class TypeUtils(t: Type) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val commas = ", " // for mkString in BundleType
        val s = t match {
          case ClockType => "Clock"
          //case UnknownType => "UnknownType"
          case UnknownType => "?"
          case t: UIntType => s"UInt${t.width.serialize}"
          case t: SIntType => s"SInt${t.width.serialize}"
          case t: BundleType => s"{${t.fields.map(_.serialize).mkString(commas)}}"
          case t: VectorType => s"${t.tpe.serialize}[${t.size}]"
        } 
        s + debug(t)
    }

    def getType(): Type = 
      t match {
        case v: VectorType => v.tpe
        case tpe: Type => UnknownType
      }

    def wipeWidth(): Type = 
      t match {
        case t: UIntType => UIntType(UnknownWidth)
        case t: SIntType => SIntType(UnknownWidth)
        case _ => t
      }
  }

  implicit class PortDirUtils(p: PortDir) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val s = p match {
        case Input => "input"
        case Output => "output"
      } 
      s + debug(p)
    }
    def toFieldDir(): FieldDir = {
      p match {
        case Input => Reverse
        case Output => Default
      }
    }
  }

  implicit class PortUtils(p: Port) {
    def serialize(implicit flags: FlagMap = FlagMap): String = 
      s"${p.dir.serialize} ${p.name} : ${p.tpe.serialize}" + debug(p)
    def getType(): Type = p.tpe
    def toField(): Field = Field(p.name, p.dir.toFieldDir, p.tpe)
  }

  implicit class ModuleUtils(m: Module) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      var s = new StringBuilder(s"module ${m.name} : ")
      withIndent {
        s ++= m.ports.map(newline ++ _.serialize).mkString
        s ++= newline ++ m.stmt.serialize
      }
      s ++= debug(m)
      s.toString
    }
  }

  implicit class CircuitUtils(c: Circuit) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      var s = new StringBuilder(s"circuit ${c.name} : ")
      withIndent { s ++= newline ++ c.modules.map(_.serialize).mkString(newline + newline) }
      s ++= newline ++ newline
      s ++= debug(c)
      s.toString
    }
  }

  private var indentLevel = 0
  private def newline = "\n" + ("  " * indentLevel)
  private def indent(): Unit = indentLevel += 1
  private def unindent() { require(indentLevel > 0); indentLevel -= 1 }
  private def withIndent(f: => Unit) { indent(); f; unindent() }

}
