// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.annotations._
import firrtl.options.{HasShellOptions, ShellOption}

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation */
case class NoDedupAnnotation(target: ModuleTarget) extends SingleTargetAnnotation[ModuleTarget] {
  def duplicate(n: ModuleTarget): NoDedupAnnotation = NoDedupAnnotation(n)
}

/** If this [[firrtl.annotations.Annotation Annotation]] exists in an [[firrtl.AnnotationSeq AnnotationSeq]],
  * then the [[firrtl.transforms.DedupModules]] transform will *NOT* be run on the circuit.
  *  - set with '--no-dedup'
  */
case object NoCircuitDedupAnnotation extends NoTargetAnnotation with HasShellOptions {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "no-dedup",
      toAnnotationSeq = _ => Seq(NoCircuitDedupAnnotation),
      helpText = "Do NOT dedup modules"
    )
  )

}
