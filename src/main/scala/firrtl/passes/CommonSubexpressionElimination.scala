// See LICENSE for license details.

package firrtl.passes

import firrtl._
import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{Dependency, PreservesAll}

object CommonSubexpressionElimination extends Pass with PreservesAll[Transform] {

  override val prerequisites = firrtl.stage.Forms.LowForm ++
    Seq( Dependency(firrtl.passes.RemoveValidIf),
         Dependency[firrtl.transforms.ConstantPropagation],
         Dependency(firrtl.passes.memlib.VerilogMemDelays),
         Dependency(firrtl.passes.SplitExpressions),
         Dependency[firrtl.transforms.CombineCats] )

  override val dependents =
    Seq( Dependency[SystemVerilogEmitter],
         Dependency[VerilogEmitter] )

  private def cse(s: Statement): Statement = {
    val expressions = collection.mutable.HashMap[MemoizedHash[Expression], String]()
    val nodes = collection.mutable.HashMap[String, Expression]()

    def eliminateNodeRef(e: Expression): Expression = e match {
      case WRef(name, tpe, kind, flow) => nodes get name match {
        case Some(expression) => expressions get expression match {
          case Some(cseName) if cseName != name =>
            WRef(cseName, tpe, kind, flow)
          case _ => e
        }
        case _ => e
      }
      case _ => e map eliminateNodeRef
    }

    def eliminateNodeRefs(s: Statement): Statement = {
      s map eliminateNodeRef match {
        case x: DefNode =>
          nodes(x.name) = x.value
          expressions.getOrElseUpdate(x.value, x.name)
          x
        case other => other map eliminateNodeRefs
      }
    }

    eliminateNodeRefs(s)
  }

  def run(c: Circuit): Circuit = {
    val modulesx = c.modules.map {
      case m: ExtModule => m
      case m: Module => Module(m.info, m.name, m.ports, cse(m.body))
    }
    Circuit(c.info, modulesx, c.main)
  }
}
