// See LICENSE for license details.

package logger.phases

import firrtl.AnnotationSeq
import firrtl.options.Phase

import logger.{LoggerOption, LogLevelAnnotation}

/** Add default logger [[Annotation]]s */
private [logger] object AddDefaults extends Phase {

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
