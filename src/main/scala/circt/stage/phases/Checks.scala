// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.stage.{CIRCTHandover, CIRCTTargetAnnotation}

import firrtl.{AnnotationSeq, EmitAllModulesAnnotation, Emitter, SystemVerilogEmitter}
import firrtl.annotations.Annotation
import firrtl.options.{Dependency, OptionsException, Phase, TargetDirAnnotation}
import firrtl.stage.OutputFileAnnotation

/** Check properties of an [[AnnotationSeq]] to look for errors before running CIRCT. */
class Checks extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq(Dependency[circt.stage.phases.AddDefaults])
  override def optionalPrerequisiteOf = Seq(Dependency[circt.stage.phases.CIRCT])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val target, outputFile, split, targetDir, handover = collection.mutable.ArrayBuffer[Annotation]()

    annotations.foreach {
      case a: OutputFileAnnotation     => outputFile += a
      case a: EmitAllModulesAnnotation => split += a
      case a: TargetDirAnnotation      => targetDir += a
      case a: CIRCTTargetAnnotation    => target += a
      case a: CIRCTHandover            => handover += a
      case _ =>
    }
    if ((split.size > 0) && (outputFile.size != 0)) {
      throw new OptionsException("Cannot specify both an OutputFileAnnotation and an EmitAllModulesAnnotation")
    }
    if ((split.size == 0) && (outputFile.size != 1)) {
      throw new OptionsException("An output file must be specified")
    }

    if ((split.size > 0) && (targetDir.size != 1)) {
      throw new OptionsException("If EmitAllModulesAnnotation is specified one TargetDirAnnotation is needed")
    }

    if (target.size != 1) {
      throw new OptionsException("Exactly one CIRCT target must be specified")
    }

    if (handover.size != 1) {
      throw new OptionsException("Exactly one handover must be specified")
    }

    annotations
  }

}
