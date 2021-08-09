// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.Mappers._
import firrtl._
import firrtl.annotations.NoTargetAnnotation
import firrtl.ir._
import firrtl.options.{Dependency, HasShellOptions, RegisteredTransform, ShellOption}

/** Indicate that CommonSubexpressionElimination should not be run */
case object NoCommonSubexpressionElimination extends NoTargetAnnotation

object CommonSubexpressionElimination extends Transform with HasShellOptions with DependencyAPIMigration {

  override def prerequisites = firrtl.stage.Forms.LowForm
  override def optionalPrerequisiteOf =
    Seq(Dependency[SystemVerilogEmitter], Dependency[VerilogEmitter])

  override def invalidates(a: Transform) = false

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-cse",
      toAnnotationSeq = _ => Seq(NoCommonSubexpressionElimination),
      helpText = "Disable common subexpression elimination"
    )
  )

  private def cse(s: Statement): Statement = {
    val expressions = collection.mutable.HashMap[MemoizedHash[Expression], String]()
    val nodes = collection.mutable.HashMap[String, Expression]()

    def eliminateNodeRef(e: Expression): Expression = e match {
      case WRef(name, tpe, kind, flow) =>
        nodes.get(name) match {
          case Some(expression) =>
            expressions.get(expression) match {
              case Some(cseName) if cseName != name =>
                WRef(cseName, tpe, kind, flow)
              case _ => e
            }
          case _ => e
        }
      case _ => e.map(eliminateNodeRef)
    }

    def eliminateNodeRefs(s: Statement): Statement = {
      s.map(eliminateNodeRef) match {
        case x: DefNode =>
          nodes(x.name) = x.value
          expressions.getOrElseUpdate(x.value, x.name)
          x
        case other => other.map(eliminateNodeRefs)
      }
    }

    eliminateNodeRefs(s)
  }

  override def execute(state: CircuitState): CircuitState =
    if (state.annotations.contains(NoCommonSubexpressionElimination))
      state
    else
      state.copy(circuit = state.circuit.copy(modules = state.circuit.modules.map({
        case m: ExtModule => m
        case m: Module    => Module(m.info, m.name, m.ports, cse(m.body))
      })))
}
