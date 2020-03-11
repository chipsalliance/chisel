// See LICENSE for license details.

package logger.phases

import firrtl.AnnotationSeq
import firrtl.annotations.Annotation
import firrtl.options.{Dependency, Phase, PreservesAll}

import logger.{LogLevelAnnotation, LogFileAnnotation, LoggerException}

import scala.collection.mutable

/** Check that an [[firrtl.AnnotationSeq AnnotationSeq]] has all necessary [[firrtl.annotations.Annotation Annotation]]s
  * for a [[Logger]] */
object Checks extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[AddDefaults])
  override val dependents = Seq.empty

  /** Ensure that an [[firrtl.AnnotationSeq AnnotationSeq]] has necessary [[Logger]] [[firrtl.annotations.Annotation
    * Annotation]]s
    * @param annotations input annotations
    * @return input annotations unmodified
    * @throws logger.LoggerException
    */
  def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val ll, lf = mutable.ListBuffer[Annotation]()
    annotations.foreach(
      _ match {
        case a: LogLevelAnnotation => ll += a
        case a: LogFileAnnotation  => lf += a
        case _                     =>         })
    if (ll.size > 1) {
      val l = ll.map{ case LogLevelAnnotation(x) => x }
      throw new LoggerException(
        s"""|At most one log level can be specified, but found '${l.mkString(", ")}' specified via:
            |    - an option or annotation: -ll, --log-level, LogLevelAnnotation""".stripMargin )}
    if (lf.size > 1) {
      throw new LoggerException(
        s"""|At most one log file can be specified, but found ${lf.size} combinations of:
            |    - an options or annotation: -ltf, --log-to-file, --log-file, LogFileAnnotation""".stripMargin )}
    annotations
  }

}
