// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
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

/** Holds the mapping from original module to the instances the original module pointed to
  * The original module target is unaffected by renaming
  * @param duplicate Instance target of what the original module now points to
  * @param original Original module
  * @param index the normalized position of the original module in the original module list, fraction between 0 and 1
  */
case class DedupedResult(original: ModuleTarget, duplicate: Option[IsModule], index: Double)
    extends MultiTargetAnnotation {
  override val targets: Seq[Seq[Target]] = Seq(Seq(original), duplicate.toList)
  override def duplicate(n: Seq[Seq[Target]]): Annotation = {
    n.toList match {
      case Seq(_, List(dup: IsModule)) => DedupedResult(original, Some(dup), index)
      case _ => DedupedResult(original, None, -1)
    }
  }
}
