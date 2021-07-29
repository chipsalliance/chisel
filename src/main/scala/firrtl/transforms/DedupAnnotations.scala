// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.Dependency
import firrtl.Utils.{groupByIntoSeq, BoolType}
import firrtl.annotations.Annotation
import scala.collection.mutable.Buffer
import firrtl.annotations.MemoryFileInlineAnnotation
import firrtl.passes.PassException
import firrtl.annotations.ReferenceTarget
import firrtl.annotations._
import firrtl.analyses.InstanceKeyGraph

import scala.collection.mutable.ArrayBuffer

object DedupAnnotationsTransform {

  final class DifferingModuleAnnotationsException private (msg: String) extends PassException(msg)
  object DifferingModuleAnnotationsException {
    def apply(left: ReferenceTarget, right: ReferenceTarget): DifferingModuleAnnotationsException = {
      val msg = s"${left.serialize} and ${right.serialize} have differing module binaries"
      new DifferingModuleAnnotationsException(msg)
    }
  }

  private case class DedupableRepr(
    dedupKey:       Any,
    deduped:        Annotation,
    original:       Annotation,
    absoluteTarget: ReferenceTarget)
  private object DedupableRepr {
    def apply(annotation: Annotation): Option[DedupableRepr] = annotation.dedup match {
      case Some((dedupKey, dedupedAnno, absoluteTarget)) =>
        Some(new DedupableRepr(dedupKey, dedupedAnno, annotation, absoluteTarget))
      case _ => None
    }
  }

  private type InstancePath = Seq[(TargetToken.Instance, TargetToken.OfModule)]

  private def checkInstanceGraph(
    module:        String,
    graph:         InstanceKeyGraph,
    absolutePaths: Seq[InstancePath]
  ): Boolean = graph.findInstancesInHierarchy(module).size == absolutePaths.size

  def dedupAnnotations(annotations: Seq[Annotation], graph: InstanceKeyGraph): Seq[Annotation] = {
    val canDedup = ArrayBuffer.empty[DedupableRepr]
    val outAnnos = ArrayBuffer.empty[Annotation]

    // Extract the annotations which can be deduplicated
    annotations.foreach { anno =>
      DedupableRepr(anno) match {
        case Some(repr) => canDedup += repr
        case None       => outAnnos += anno
      }
    }

    // Partition the dedupable annotations into groups that *should* deduplicate into the same annotation
    val shouldDedup: Seq[(Any, Seq[DedupableRepr])] = groupByIntoSeq(canDedup)(_.dedupKey)
    shouldDedup.foreach {
      case ((target: ReferenceTarget, _), dedupableAnnos) =>
        val originalAnnos = dedupableAnnos.map(_.original)
        val uniqueDedupedAnnos = dedupableAnnos.map(_.deduped).distinct
        // TODO: Extend this to support multi-target annotations
        val instancePaths = dedupableAnnos.map(_.absoluteTarget.path).toSeq
        // The annotation deduplication is only legal if it applies to *all* instances of a
        // deduplicated module -- requires an instance graph check
        if (uniqueDedupedAnnos.size == 1 && checkInstanceGraph(target.encapsulatingModule, graph, instancePaths))
          outAnnos += uniqueDedupedAnnos.head
        else
          outAnnos ++= originalAnnos
    }

    outAnnos.toSeq
  }
}

/** Deduplicates memory annotations
  */
class DedupAnnotationsTransform extends Transform with DependencyAPIMigration {

  override def prerequisites = Nil

  override def optionalPrerequisites = Nil

  override def optionalPrerequisiteOf = Nil

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = CircuitState(
    state.circuit,
    state.form,
    DedupAnnotationsTransform.dedupAnnotations(state.annotations.underlying, InstanceKeyGraph(state.circuit)),
    state.renames
  )
}
