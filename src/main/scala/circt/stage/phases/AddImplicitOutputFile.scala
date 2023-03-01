// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.stage.SplitVerilog

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, Viewer}
import firrtl.stage.{FirrtlOptions, OutputFileAnnotation}

/** [[firrtl.options.Phase Phase]] that adds an [[firrtl.stage.OutputFileAnnotation OutputFileAnnotation]] if one does
  * not already exist.
  *
  * To determine the [[firrtl.stage.OutputFileAnnotation OutputFileAnnotation]], the following precedence is
  * used. Whichever happens first succeeds:
  *  - Do nothing if an [[firrtl.stage.OutputFileAnnotation OutputFileAnnotation]] "--split-verilog" was specified
  *  - Use the main in the first discovered [[firrtl.stage.FirrtlCircuitAnnotation FirrtlCircuitAnnotation]] (see note
  *    below)
  *  - Use "a"
  */
class AddImplicitOutputFile extends Phase {

  override def prerequisites = Seq.empty

  override def optionalPrerequisiteOf = Seq.empty

  override def invalidates(a: Phase) = false

  /** Add an [[firrtl.stage.OutputFileAnnotation OutputFileAnnotation]] to an [[firrtl.AnnotationSeq AnnotationSeq]] */
  def transform(annotations: AnnotationSeq): AnnotationSeq =
    annotations.collectFirst { case _: OutputFileAnnotation | SplitVerilog => annotations }.getOrElse {
      val topName = Viewer[FirrtlOptions]
        .view(annotations)
        .firrtlCircuit
        .map(_.main)
        .getOrElse("a")
      OutputFileAnnotation(topName) +: annotations
    }
}
