// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.stage.CIRCTTargetAnnotation

import firrtl.{
  AnnotationSeq,
  Emitter,
  SystemVerilogEmitter
}
import firrtl.annotations.Annotation
import firrtl.options.{
  Dependency,
  OptionsException,
  Phase
}
import firrtl.stage.{
  RunFirrtlTransformAnnotation,
  OutputFileAnnotation
}

/** Check properties of an [[AnnotationSeq]] to look for errors before running CIRCT. */
class Checks extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[circt.stage.phases.CIRCT])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val transforms, outputFile, target = collection.mutable.ArrayBuffer[Annotation]()

    annotations.foreach {
      case a @ RunFirrtlTransformAnnotation(_: Emitter) =>
      case a: RunFirrtlTransformAnnotation => transforms += a
      case a: OutputFileAnnotation => outputFile += a
      case a: CIRCTTargetAnnotation => target += a
      case _ =>
    }

    if (!transforms.isEmpty) {
      throw new OptionsException("CIRCT does not support any custom transforms")
    }

    if (outputFile.size != 1) {
      throw new OptionsException("An output file must be specified")
    }

    if (target.size != 1) {
      throw new OptionsException("Exactly one CIRCT target must be specified")
    }

    annotations
  }

}
