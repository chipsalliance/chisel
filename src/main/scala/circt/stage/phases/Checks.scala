// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.stage.{CIRCTTargetAnnotation, SplitVerilog}

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{Dependency, OptionsException, Phase, TargetDirAnnotation}
import firrtl.stage.OutputFileAnnotation

/** Check properties of an [[firrtl.AnnotationSeq!]] to look for errors before running CIRCT. */
class Checks extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[circt.stage.phases.CIRCT])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val target, outputFile, split, targetDir = collection.mutable.ArrayBuffer[Annotation]()

    annotations.foreach {
      case a: OutputFileAnnotation => outputFile += a
      case a @ SplitVerilog => split += a
      case a: TargetDirAnnotation   => targetDir += a
      case a: CIRCTTargetAnnotation => target += a
      case _ =>
    }

    if ((split.size > 0) && (outputFile.size != 0)) {
      throw new OptionsException("Cannot specify both SplitVerilog and an EmitAllModulesAnnotation")
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

    annotations
  }

}
