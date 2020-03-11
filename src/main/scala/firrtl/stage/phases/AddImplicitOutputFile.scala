// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.{AnnotationSeq, EmitAllModulesAnnotation}
import firrtl.options.{Dependency, Phase, PreservesAll, Viewer}
import firrtl.stage.{FirrtlOptions, OutputFileAnnotation}

/** [[firrtl.options.Phase Phase]] that adds an [[OutputFileAnnotation]] if one does not already exist.
  *
  * To determine the [[OutputFileAnnotation]], the following precedence is used. Whichever happens first succeeds:
  *  - Do nothing if an [[OutputFileAnnotation]] or [[EmitAllModulesAnnotation]] exist
  *  - Use the main in the first discovered [[firrtl.stage.FirrtlCircuitAnnotation FirrtlCircuitAnnotation]] (see note
  *    below)
  *  - Use "a"
  *
  * The file suffix may or may not be specified, but this may be arbitrarily changed by the [[Emitter]].
  *
  * @note This [[firrtl.options.Phase Phase]] has a dependency on [[AddCircuit]]. Only a
  * [[firrtl.stage.FirrtlCircuitAnnotation FirrtlCircuitAnnotation]] will be used to implicitly set the
  * [[OutputFileAnnotation]] (not other [[firrtl.stage.CircuitOption CircuitOption]] subclasses).
  */
class AddImplicitOutputFile extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[AddCircuit])

  override val dependents = Seq.empty

  /** Add an [[OutputFileAnnotation]] to an [[AnnotationSeq]] */
  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations
      .collectFirst { case _: OutputFileAnnotation | _: EmitAllModulesAnnotation => annotations }
      .getOrElse {
        val topName = Viewer[FirrtlOptions].view(annotations)
          .firrtlCircuit
          .map(_.main)
          .getOrElse("a")
        OutputFileAnnotation(topName) +: annotations
      }
}
