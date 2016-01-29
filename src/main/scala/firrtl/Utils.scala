// Utility functions for FIRRTL IR

/* TODO
 *  - Adopt style more similar to Chisel3 Emitter?
 *  - Find way to have generic map function instead of mapE and mapS under Stmt implicits
 */

/* TODO Richard
 *  - add new IR nodes to all Util functions
 */

package firrtl

import scala.collection.mutable.StringBuilder
import java.io.PrintWriter
import PrimOps._
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
        case e: Expression => e.getType
        case s: Stmt => s.getType
        //case f: Field => f.getType
        case t: Type => t.getType
        case p: Port => p.getType
        case _ => UnknownType
      }
  }

  implicit class PrimOpUtils(op: PrimOp) {
    def serialize(implicit flags: FlagMap = FlagMap): String = op.getString
  }

  implicit class ExpUtils(exp: Expression) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val ret = exp match {
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
      } 
      ret + debug(exp)
    }

    def map(f: Expression => Expression): Expression = 
      exp match {
        case s: SubField => SubField(f(s.exp), s.name, s.tpe)
        case s: SubIndex => SubIndex(f(s.exp), s.value, s.tpe)
        case s: SubAccess => SubAccess(f(s.exp), f(s.index), s.tpe)
        case m: Mux => Mux(f(m.cond), f(m.tval), f(m.fval), m.tpe)
        case v: ValidIf => ValidIf(f(v.cond), f(v.value), v.tpe)
        case p: DoPrim => DoPrim(p.op, p.args.map(f), p.consts, p.tpe)
        case e: Expression => e
      }

    def getType(): Type = {
      exp match {
        case v: UIntValue => UIntType(UnknownWidth)
        case v: SIntValue => SIntType(UnknownWidth)
        case r: Ref => r.tpe
        case s: SubField => s.tpe
        case s: SubIndex => s.tpe
        case s: SubAccess => s.tpe
        case p: DoPrim => p.tpe
        case m: Mux => m.tpe
        case v: ValidIf => v.tpe
      }
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
          case w: Conditionally => Conditionally(w.info, w.pred, f(w.conseq), f(w.alt))
          case b: Begin => Begin(b.stmts.map(f))
          case s: Stmt => s
        }
    }
    implicit def forExp(f: Expression => Expression) = new StmtMagnet {
      override def map(stmt: Stmt): Stmt =
        stmt match { 
          case r: DefRegister => DefRegister(r.info, r.name, r.tpe, f(r.clock), f(r.reset), f(r.init))
          case n: DefNode => DefNode(n.info, n.name, f(n.value))
          case c: Connect => Connect(c.info, f(c.loc), f(c.exp))
          case b: BulkConnect => BulkConnect(b.info, f(b.loc), f(b.exp))
          case w: Conditionally => Conditionally(w.info, f(w.pred), w.conseq, w.alt)
          case i: IsInvalid => IsInvalid(i.info, f(i.exp))
          case s: Stop => Stop(s.info, s.ret, f(s.clk), f(s.en))
          case p: Print => Print(p.info, p.string, p.args.map(f), f(p.clk), f(p.en))
          case s: Stmt => s 
        }
    }
  }

  implicit class StmtUtils(stmt: Stmt) {
    def serialize(implicit flags: FlagMap = FlagMap): String =
    {
      var ret = stmt match {
        case w: DefWire => s"wire ${w.name} : ${w.tpe.serialize}"
        case r: DefRegister => 
          val str = new StringBuilder(s"reg ${r.name} : ${r.tpe.serialize}, ${r.clock.serialize} with : ")
          withIndent {
            str ++= newline + s"reset => (${r.reset.serialize}, ${r.init.serialize})"
          }
          str
        case i: DefInstance => s"inst ${i.name} of ${i.module}"
        case m: DefMemory => {
          val str = new StringBuilder(s"mem ${m.name} : " + newline)
          withIndent {
            str ++= s"data-type => ${m.dataType.serialize}" + newline +
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
        case n: DefNode => s"node ${n.name} = ${n.value.serialize}"
        case c: Connect => s"${c.loc.serialize} <= ${c.exp.serialize}"
        case b: BulkConnect => s"${b.loc.serialize} <- ${b.exp.serialize}"
        case w: Conditionally => {
          var str = new StringBuilder(s"when ${w.pred.serialize} : ")
          withIndent { str ++= w.conseq.serialize }
          if( w.alt != Empty ) {
            str ++= newline + "else :"
            withIndent { str ++= w.alt.serialize }
          }
          str.result
        }
        case b: Begin => {
          val s = new StringBuilder
          b.stmts.foreach { s ++= newline ++ _.serialize }
          s.result + debug(b)
        } 
        case i: IsInvalid => s"${i.exp.serialize} is invalid"
        case s: Stop => s"stop(${s.clk.serialize}, ${s.en.serialize}, ${s.ret})"
        case p: Print => s"printf(${p.clk.serialize}, ${p.en.serialize}, ${p.string}" + 
                         (if (p.args.nonEmpty) p.args.map(_.serialize).mkString(", ", ", ", "") else "") + ")"
        case Empty => "skip"
      } 
      ret + debug(stmt)
    }

    // Using implicit types to allow overloading of function type to map, see StmtMagnet above
    def map[T](f: T => T)(implicit magnet: (T => T) => StmtMagnet): Stmt = magnet(f).map(stmt)
    
    def getType(): Type =
      stmt match {
        case s: DefWire    => s.tpe
        case s: DefRegister => s.tpe
        case s: DefMemory  => s.dataType
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

  implicit class FlipUtils(f: Flip) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val s = f match {
        case Reverse => "flip "
        case Default => ""
      } 
      s + debug(f)
    }
    def flip(): Flip = {
      f match {
        case Reverse => Default
        case Default => Reverse
      }
    }
        
    def toDirection(): Direction = {
      f match {
        case Default => Output
        case Reverse => Input
      }
    }
  }

  implicit class FieldUtils(field: Field) {
    def serialize(implicit flags: FlagMap = FlagMap): String = 
      s"${field.flip.serialize}${field.name} : ${field.tpe.serialize}" + debug(field)
    def flip(): Field = Field(field.name, field.flip.flip, field.tpe)

    def getType(): Type = field.tpe
    def toPort(): Port = Port(NoInfo, field.name, field.flip.toDirection, field.tpe)
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
          case t: BundleType => s"{ ${t.fields.map(_.serialize).mkString(commas)}}"
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

  implicit class DirectionUtils(d: Direction) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      val s = d match {
        case Input => "input"
        case Output => "output"
      } 
      s + debug(d)
    }
    def toFlip(): Flip = {
      d match {
        case Input => Reverse
        case Output => Default
      }
    }
  }

  implicit class PortUtils(p: Port) {
    def serialize(implicit flags: FlagMap = FlagMap): String = 
      s"${p.dir.serialize} ${p.name} : ${p.tpe.serialize}" + debug(p)
    def getType(): Type = p.tpe
    def toField(): Field = Field(p.name, p.dir.toFlip, p.tpe)
  }

  implicit class ModuleUtils(m: Module) {
    def serialize(implicit flags: FlagMap = FlagMap): String = {
      var s = new StringBuilder(s"module ${m.name} : ")
      withIndent {
        s ++= m.ports.map(newline ++ _.serialize).mkString
        s ++= m.stmt.serialize
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
