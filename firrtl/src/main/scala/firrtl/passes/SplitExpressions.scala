// SPDX-License-Identifier: Apache-2.0

package firrtl
package passes

import firrtl.{SystemVerilogEmitter, Transform, VerilogEmitter}
import firrtl.ir._
import firrtl.options.Dependency
import firrtl.Mappers._
import firrtl.Utils.{flow, get_info, kind}
import firrtl.transforms.InlineBooleanExpressions

// Datastructures
import scala.collection.mutable

// Splits compound expressions into simple expressions
//  and named intermediate nodes
object SplitExpressions extends Pass {

  override def prerequisites = firrtl.stage.Forms.LowForm
  override def optionalPrerequisiteOf =
    Seq(Dependency[SystemVerilogEmitter], Dependency[VerilogEmitter])

  override def invalidates(a: Transform) = a match {
    case _: InlineBooleanExpressions => true // SplitExpressions undoes the inlining!
    case _ => false
  }

  private def isSignedArithmetic(e: Expression): Boolean = e match {
    case DoPrim(PrimOps.Add, _, _, _: SIntType) => true
    case DoPrim(PrimOps.Sub, _, _, _: SIntType) => true
    case DoPrim(PrimOps.Mul, _, _, _: SIntType) => true
    case DoPrim(PrimOps.Div, _, _, _: SIntType) => true
    case _ => false
  }

  private def onModule(m: Module): Module = {
    val namespace = Namespace(m)
    def onStmt(s: Statement): Statement = {
      val v = mutable.ArrayBuffer[Statement]()
      // Splits current expression if needed
      // Adds named temporaries to v
      def split(e: Expression): Expression = e match {
        case e: DoPrim =>
          val name = namespace.newTemp
          v += DefNode(get_info(s), name, e)
          WRef(name, e.tpe, kind(e), flow(e))
        case e: Mux =>
          val name = namespace.newTemp
          v += DefNode(get_info(s), name, e)
          WRef(name, e.tpe, kind(e), flow(e))
        case e: ValidIf =>
          val name = namespace.newTemp
          v += DefNode(get_info(s), name, e)
          WRef(name, e.tpe, kind(e), flow(e))
        case _ => e
      }

      // Recursive. Splits compound nodes
      def onExp(e: Expression): Expression =
        e.map(onExp) match {
          case ex: DoPrim => ex.map(split)
          // Arguably we should be splitting all Mux expressions but this has a negative impact on
          // Verilog, instead this is a focused fix for
          // https://github.com/chipsalliance/firrtl/issues/2439
          case ex: Mux if isSignedArithmetic(ex.tval) || isSignedArithmetic(ex.fval) => ex.map(split)
          case ex => ex
        }

      s.map(onExp) match {
        case x: Block => x.map(onStmt)
        case EmptyStmt => EmptyStmt
        case x =>
          v += x
          v.size match {
            case 1 => v.head
            case _ => Block(v.toSeq)
          }
      }
    }
    Module(m.info, m.name, m.ports, onStmt(m.body))
  }
  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: Module    => onModule(m)
      case m: ExtModule => m
    }
    Circuit(c.info, modulesx, c.main)
  }
}
