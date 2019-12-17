// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{AnnotationSeq, CircuitState, RenameMap, Transform, Utils}
import firrtl.annotations.{Annotation, DeletedAnnotation}
import firrtl.options.Translator

import scala.collection.mutable

class UpdateAnnotations(val underlying: Transform) extends Transform with WrappedTransform
    with Translator[CircuitState, (CircuitState, CircuitState)] {

  override def execute(c: CircuitState): CircuitState = underlying.execute(c)

  def aToB(a: CircuitState): (CircuitState, CircuitState) = (a, a)

  def bToA(b: (CircuitState, CircuitState)): CircuitState = {
    val (state, result) = (b._1, b._2)

    val remappedAnnotations = propagateAnnotations(state.annotations, result.annotations, result.renames)

    logger.info(s"Form: ${result.form}")

    logger.debug(s"Annotations:")
    remappedAnnotations.foreach( a => logger.debug(a.serialize) )

    logger.trace(s"Circuit:\n${result.circuit.serialize}")
    logger.info(s"======== Finished Transform $name ========\n")

    CircuitState(result.circuit, result.form, remappedAnnotations, None)
  }

  def internalTransform(b: (CircuitState, CircuitState)): (CircuitState, CircuitState) = {
    logger.info(s"======== Starting Transform $name ========")

    /* @todo: prepare should likely be factored out of this */
    val (timeMillis, result) = Utils.time { execute( trueUnderlying.prepare(b._2) ) }

    logger.info(s"""----------------------------${"-" * name.size}---------\n""")
    logger.info(f"Time: $timeMillis%.1f ms")

    (b._1, result)
  }

  /** Propagate annotations and update their names.
    *
    * @param inAnno input AnnotationSeq
    * @param resAnno result AnnotationSeq
    * @param renameOpt result RenameMap
    * @return the updated annotations
    */
  private[firrtl] def propagateAnnotations(
      inAnno: AnnotationSeq,
      resAnno: AnnotationSeq,
      renameOpt: Option[RenameMap]): AnnotationSeq = {
    val newAnnotations = {
      val inSet = mutable.LinkedHashSet() ++ inAnno
      val resSet = mutable.LinkedHashSet() ++ resAnno
      val deleted = (inSet -- resSet).map {
        case DeletedAnnotation(xFormName, delAnno) => DeletedAnnotation(s"$xFormName+$name", delAnno)
        case anno => DeletedAnnotation(name, anno)
      }
      val created = resSet -- inSet
      val unchanged = resSet & inSet
      (deleted ++ created ++ unchanged)
    }

    // For each annotation, rename all annotations.
    val renames = renameOpt.getOrElse(RenameMap())
    val remapped2original = mutable.LinkedHashMap[Annotation, mutable.LinkedHashSet[Annotation]]()
    val keysOfNote = mutable.LinkedHashSet[Annotation]()
    val finalAnnotations = newAnnotations.flatMap { anno =>
      val remappedAnnos = anno.update(renames)
      remappedAnnos.foreach { remapped =>
        val set = remapped2original.getOrElseUpdate(remapped, mutable.LinkedHashSet.empty[Annotation])
        set += anno
        if(set.size > 1) keysOfNote += remapped
      }
      remappedAnnos
    }.toSeq
    keysOfNote.foreach { key =>
      logger.debug(s"""The following original annotations are renamed to the same new annotation.""")
      logger.debug(s"""Original Annotations:\n  ${remapped2original(key).mkString("\n  ")}""")
      logger.debug(s"""New Annotation:\n  $key""")
    }
    finalAnnotations
  }
}

object UpdateAnnotations {

  def apply(a: Transform): UpdateAnnotations = new UpdateAnnotations(a)

}
