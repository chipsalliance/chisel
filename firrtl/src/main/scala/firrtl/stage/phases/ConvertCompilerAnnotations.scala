// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.phases

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, OptionsException, Phase}
import firrtl.stage.{CompilerAnnotation, RunFirrtlTransformAnnotation}

private[firrtl] class ConvertCompilerAnnotations extends Phase {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq(Dependency[AddDefaults], Dependency[Checks])
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    annotations.collect {
      case a: CompilerAnnotation => a
    } match {
      case a if a.size > 1 =>
        throw new OptionsException(
          s"Zero or one CompilerAnnotation may be specified, but found '${a.mkString(", ")}'.".stripMargin
        )
      case _ =>
    }
    annotations.map {
      case CompilerAnnotation(a) =>
        val suggestion = s"RunFirrtlTransformAnnotation(new ${a.emitter.getClass.getName})"
        logger.warn(s"CompilerAnnotation is deprecated since FIRRTL 1.4.0. Please use '$suggestion' instead.")
        RunFirrtlTransformAnnotation(a.emitter)
      case a => a
    }
  }

}
