// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms.formal

import firrtl.ir.{Circuit, Formal, Statement, Verification}
import firrtl.stage.TransformManager.TransformDependency
import firrtl.{CircuitState, DependencyAPIMigration, Transform}
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{PreservesAll, RegisteredTransform, ShellOption}

/**
  * Assert Submodule Assumptions
  *
  * Converts `assume` statements to `assert` statements in all modules except
  * the top module being compiled. This avoids a class of bugs in which an
  * overly restrictive assume in a child module can prevent the model checker
  * from searching valid inputs and states in the parent module.
  */
class AssertSubmoduleAssumptions
    extends Transform
    with RegisteredTransform
    with DependencyAPIMigration
    with PreservesAll[Transform] {

  override def prerequisites:         Seq[TransformDependency] = Seq.empty
  override def optionalPrerequisites: Seq[TransformDependency] = Seq.empty
  override def optionalPrerequisiteOf: Seq[TransformDependency] =
    firrtl.stage.Forms.MidEmitters

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-asa",
      toAnnotationSeq = (_: Unit) => Seq(DontAssertSubmoduleAssumptionsAnnotation),
      helpText = "Disable assert submodule assumptions"
    )
  )

  def assertAssumption(s: Statement): Statement = s match {
    case v: Verification if v.op == Formal.Assume => v.withOp(Formal.Assert)
    case t => t.mapStmt(assertAssumption)
  }

  def run(c: Circuit): Circuit = {
    c.mapModule(mod => {
      if (mod.name != c.main) {
        mod.mapStmt(assertAssumption)
      } else {
        mod
      }
    })
  }

  def execute(state: CircuitState): CircuitState = {
    val noASA = state.annotations.contains(DontAssertSubmoduleAssumptionsAnnotation)
    if (noASA) {
      logger.info("Skipping assert submodule assumptions")
      state
    } else {
      state.copy(circuit = run(state.circuit))
    }
  }
}

case object AssertSubmoduleAssumptionsAnnotation extends NoTargetAnnotation {
  val transform = new AssertSubmoduleAssumptions
}

case object DontAssertSubmoduleAssumptionsAnnotation extends NoTargetAnnotation
