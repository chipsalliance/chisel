// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.stage._

import firrtl.{AnnotationSeq, Parser}
import firrtl.options.{Dependency, Phase, PhasePrerequisiteException, PreservesAll}

/** [[firrtl.options.Phase Phase]] that expands [[FirrtlFileAnnotation]]/[[FirrtlSourceAnnotation]] into
  * [[FirrtlCircuitAnnotation]]s and deletes the originals. This is part of the preprocessing done on an input
  * [[AnnotationSeq]] by [[FirrtlStage]].
  *
  * The types of possible annotations are handled in the following ways:
  *  - [[FirrtlFileAnnotation]]s are read as Protocol Buffers if the file extension ends in `.pb`. Otherwise, these are
  *    assumed to be raw FIRRTL text and is sent to the [[Parser]]. The original [[FirrtlFileAnnotation]] is deleted.
  *  - [[FirrtlSourceAnnotation]]s are run through the [[Parser]]. The original [[FirrtlSourceAnnotation]] is deleted.
  *  - [[FirrtlCircuitAnnotation]]s are left untouched (along with all other annotations).
  *
  * If a [[Parser]] is used, its [[Parser.InfoMode InfoMode]] is read from a ''mandatory'' [[InfoModeAnnotation]]. If
  * using an [[Parser.InfoMode InfoMode]] that expects a filename, the filename is used for [[FirrtlFileAnnotation]]s
  * and `[anonymous source]` is used for [[FirrtlSourceAnnotation]]s.
  *
  * @note '''This must be run after [[AddDefaults]] as this [[firrtl.options.Phase Phase]] depends on the existence of
  * an [[InfoModeAnnotation]].'''.
  * @define infoModeException firrtl.options.PhasePrerequisiteException if no [[InfoModeAnnotation]] is present
  */
class AddCircuit extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[AddDefaults], Dependency[Checks])

  override val dependents = Seq.empty

  /** Extract the info mode from an [[AnnotationSeq]] or use the default info mode if no annotation exists
    * @param annotations some annotations
    * @return the info mode
    * @throws $infoModeException
    */
  private def infoMode(annotations: AnnotationSeq): Parser.InfoMode = {
    val infoModeAnnotation = annotations
      .collectFirst{ case a: InfoModeAnnotation => a }
      .getOrElse { throw new PhasePrerequisiteException(
                    "An InfoModeAnnotation must be present (did you forget to run AddDefaults?)") }
    val infoSource = annotations.collectFirst{
      case FirrtlFileAnnotation(f) => f
      case _: FirrtlSourceAnnotation => "anonymous source"
    }.getOrElse("not defined")

    infoModeAnnotation.toInfoMode(Some(infoSource))
  }

  /** Convert [[FirrtlFileAnnotation]]/[[FirrtlSourceAnnotation]] into [[FirrtlCircuitAnnotation]] and delete originals
    * @throws $infoModeException
    */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    lazy val info = infoMode(annotations)
    annotations.map {
      case a: CircuitOption => a.toCircuit(info)
      case a                => a
    }
  }

}
