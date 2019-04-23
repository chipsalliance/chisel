// See LICENSE for license details.

package firrtl.options.phases

import firrtl.AnnotationSeq
import firrtl.annotations.DeletedAnnotation
import firrtl.options.{Phase, Translator}

import scala.collection.mutable

/** Wrap a [[firrtl.options.Phase Phase]] such that any [[firrtl.annotations.Annotation Annotation]] removed by the
  * wrapped [[firrtl.options.Phase Phase]] will be added as [[firrtl.annotations.DeletedAnnotation DeletedAnnotation]]s.
  * @param p a [[firrtl.options.Phase Phase]] to wrap
  */
class DeletedWrapper(p: Phase) extends Phase with Translator[AnnotationSeq, (AnnotationSeq, AnnotationSeq)] {

  override lazy val name: String = p.name

  def aToB(a: AnnotationSeq): (AnnotationSeq, AnnotationSeq) = (a, a)

  def bToA(b: (AnnotationSeq, AnnotationSeq)): AnnotationSeq = {

    val (in, out) = (mutable.LinkedHashSet() ++ b._1, mutable.LinkedHashSet() ++ b._2)

    (in -- out).map {
      case DeletedAnnotation(n, a) => DeletedAnnotation(s"$n+$name", a)
      case a                       => DeletedAnnotation(name, a)
    }.toSeq ++ b._2

  }

  def internalTransform(b: (AnnotationSeq, AnnotationSeq)): (AnnotationSeq, AnnotationSeq) = (b._1, p.transform(b._2))

}

object DeletedWrapper {

  /** Wrap a [[firrtl.options.Phase Phase]] in a [[DeletedWrapper]]
    * @param p a [[firrtl.options.Phase Phase]] to wrap
    */
  def apply(p: Phase): DeletedWrapper = new DeletedWrapper(p)

}
