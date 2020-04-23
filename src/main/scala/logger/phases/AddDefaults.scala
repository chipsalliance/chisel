// See LICENSE for license details.

package logger.phases

import firrtl.AnnotationSeq
import firrtl.options.{Phase, PreservesAll}

import logger.{LoggerOption, LogLevelAnnotation}

/** Add default logger [[Annotation]]s */
private [logger] class AddDefaults extends Phase with PreservesAll[Phase] {

  override def prerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty

  /** Add missing default [[Logger]] [[Annotation]]s to an [[AnnotationSeq]]
    * @param annotations input annotations
    * @return output annotations with defaults
    */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    var ll = true
    annotations.collect{ case a: LoggerOption => a }.map{
      case _: LogLevelAnnotation => ll = false
      case _                     =>
    }
    annotations ++
      (if (ll) Seq(LogLevelAnnotation()) else Seq() )
  }

}
