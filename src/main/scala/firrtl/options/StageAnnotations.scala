// See LICENSE for license details

package firrtl.options

import firrtl.AnnotationSeq
import firrtl.annotations.NoTargetAnnotation

import scopt.OptionParser

sealed trait StageOption extends HasScoptOptions

/** Holds a filename containing one or more [[annotations.Annotation]] to be read
  *  - this is not stored in [[FirrtlExecutionOptions]]
  *  - set with `-faf/--annotation-file`
  * @param value input annotation filename
  */
case class InputAnnotationFileAnnotation(value: String) extends NoTargetAnnotation with StageOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("annotation-file")
    .abbr("faf")
    .unbounded()
    .valueName("<input-anno-file>")
    .action( (x, c) => c :+ InputAnnotationFileAnnotation(x) )
    .text("Used to specify annotation file")
}

object InputAnnotationFileAnnotation {
  private [firrtl] def apply(): InputAnnotationFileAnnotation = InputAnnotationFileAnnotation("")
}

/** Holds the name of the target directory
  *  - set with `-td/--target-dir`
  *  - if unset, a [[TargetDirAnnotation]] will be generated with the
  * @param value target directory name
  */
case class TargetDirAnnotation(dir: String = ".") extends NoTargetAnnotation with StageOption {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("target-dir")
    .abbr("td")
    .valueName("<target-directory>")
    .action( (x, c) => c ++ Seq(TargetDirAnnotation(x)) )
    .unbounded() // See [Note 1]
    .text(s"Work directory for intermediate files/blackboxes, default is '.' (current directory)")
}
