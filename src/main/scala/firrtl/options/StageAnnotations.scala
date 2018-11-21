// See LICENSE for license details

package firrtl.options

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}

import scopt.OptionParser

sealed trait StageOption { this: Annotation => }

/** An annotation that should not be serialized automatically [[phases.WriteOutputAnnotations WriteOutputAnnotations]].
  * This usually means that this is an annotation that is used only internally to a [[Stage]].
  */
trait Unserializable { this: Annotation => }

/** Holds the name of the target directory
  *  - set with `-td/--target-dir`
  *  - if unset, a [[TargetDirAnnotation]] will be generated with the
  * @param value target directory name
  */
case class TargetDirAnnotation(directory: String = ".") extends NoTargetAnnotation with StageOption

object TargetDirAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("target-dir")
    .abbr("td")
    .valueName("<target-directory>")
    .action( (x, c) => TargetDirAnnotation(x) +: c )
    .unbounded() // See [Note 1]
    .text(s"Work directory for intermediate files/blackboxes, default is '.' (current directory)")
}

/** Additional arguments
  *  - set with any trailing option on the command line
  * @param value one [[scala.Predef.String String]] argument
  */
case class ProgramArgsAnnotation(arg: String) extends NoTargetAnnotation with StageOption

object ProgramArgsAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.arg[String]("<arg>...")
    .unbounded()
    .optional()
    .action( (x, c) => ProgramArgsAnnotation(x) +: c )
    .text("optional unbounded args")
}

/** Holds a filename containing one or more [[annotations.Annotation]] to be read
  *  - this is not stored in [[FirrtlExecutionOptions]]
  *  - set with `-faf/--annotation-file`
  * @param value input annotation filename
  */
case class InputAnnotationFileAnnotation(file: String) extends NoTargetAnnotation with StageOption

object InputAnnotationFileAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("annotation-file")
    .abbr("faf")
    .unbounded()
    .valueName("<input-anno-file>")
    .action( (x, c) => InputAnnotationFileAnnotation(x) +: c )
    .text("Used to specify annotation file")
}

/** An explicit output _annotation_ file to write to
  *  - set with `-foaf/--output-annotation-file`
  * @param value output annotation filename
  */
case class OutputAnnotationFileAnnotation(file: String) extends NoTargetAnnotation with StageOption

object OutputAnnotationFileAnnotation extends HasScoptOptions {
  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p.opt[String]("output-annotation-file")
    .abbr("foaf")
    .valueName ("<output-anno-file>")
    .action( (x, c) => OutputAnnotationFileAnnotation(x) +: c )
    .unbounded()
    .text("use this to set the annotation output file")
}

/** If this [[firrtl.annotations.Annotation Annotation]] exists in an [[firrtl.AnnotationSeq AnnotationSeq]], then a
  * [[firrtl.options.phase.WriteOutputAnnotations WriteOutputAnnotations]] will include
  * [[firrtl.annotations.DeletedAnnotation DeletedAnnotation]]s when it writes to a file.
  *  - set with '--write-deleted'
  */
case object WriteDeletedAnnotation extends NoTargetAnnotation with StageOption with HasScoptOptions {

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .opt[Unit]("write-deleted")
    .unbounded()
    .action( (_, c) => WriteDeletedAnnotation +: c )
    .text("Include deleted annotations in the output annotation file")

}
