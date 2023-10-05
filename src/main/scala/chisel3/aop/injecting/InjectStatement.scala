// SPDX-License-Identifier: Apache-2.0

package chisel3.aop.injecting

import chisel3.stage.phases.AspectPhase
import firrtl.annotations.{Annotation, ModuleTarget, NoTargetAnnotation, SingleTargetAnnotation}

/** Contains all information needed to inject statements into a module
  *
  * Generated when a [[InjectingAspect]] is consumed by a `AspectPhase`
  * Consumed by [[InjectingPhase]]
  *
  * @param module Module to inject code into at the end of the module
  * @param s Statements to inject
  * @param modules Additional modules that may be instantiated by s
  * @param annotations Additional annotations that should be passed down compiler
  */
case class InjectStatement(
  module:      ModuleTarget,
  s:           firrtl.ir.Statement,
  modules:     Seq[firrtl.ir.DefModule],
  annotations: Seq[Annotation])
    extends SingleTargetAnnotation[ModuleTarget] {
  val target: ModuleTarget = module
  override def duplicate(n: ModuleTarget): Annotation = this.copy(module = n)
}
