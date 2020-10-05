// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import firrtl.options.{
  Dependency,
  Phase,
  PhaseManager
}
import firrtl.options.phases.DeletedWrapper

private[chisel3] class ChiselPhase extends PhaseManager(ChiselPhase.targets) {

  override val wrappers = Seq( (a: Phase) => DeletedWrapper(a) )

}

private[chisel3] object ChiselPhase {

  val targets: Seq[PhaseManager.PhaseDependency] =
    Seq( Dependency[chisel3.stage.phases.Checks],
         Dependency[chisel3.stage.phases.AddImplicitOutputFile],
         Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
         Dependency[chisel3.stage.phases.MaybeAspectPhase],
         Dependency[chisel3.stage.phases.AddSerializationAnnotations],
         Dependency[chisel3.stage.phases.Convert],
         Dependency[chisel3.stage.phases.MaybeFirrtlStage] )

}
