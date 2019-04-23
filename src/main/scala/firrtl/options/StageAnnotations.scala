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

object TargetDirAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "target-dir",
      toAnnotationSeq = (a: String) => Seq(TargetDirAnnotation(a)),
      helpText = "Work directory (default: '.')",
      shortOption = Some("td"),
      helpValueName = Some("<directory>") ) )

}

/** Additional arguments
  *  - set with any trailing option on the command line
  * @param value one [[scala.Predef.String String]] argument
  */
case class ProgramArgsAnnotation(arg: String) extends NoTargetAnnotation with StageOption

object ProgramArgsAnnotation {

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

object InputAnnotationFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "annotation-file",
      toAnnotationSeq = (a: String) => Seq(InputAnnotationFileAnnotation(a)),
      helpText = "An input annotation file",
      shortOption = Some("faf"),
      helpValueName = Some("<file>") ) )

}

/** An explicit output _annotation_ file to write to
  *  - set with `-foaf/--output-annotation-file`
  * @param value output annotation filename
  */
case class OutputAnnotationFileAnnotation(file: String) extends NoTargetAnnotation with StageOption

object OutputAnnotationFileAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "output-annotation-file",
      toAnnotationSeq = (a: String) => Seq(OutputAnnotationFileAnnotation(a)),
      helpText = "An output annotation file",
      shortOption = Some("foaf"),
      helpValueName = Some("<file>") ) )

}

/** If this [[firrtl.annotations.Annotation Annotation]] exists in an [[firrtl.AnnotationSeq AnnotationSeq]], then a
  * [[firrtl.options.phase.WriteOutputAnnotations WriteOutputAnnotations]] will include
  * [[firrtl.annotations.DeletedAnnotation DeletedAnnotation]]s when it writes to a file.
  *  - set with '--write-deleted'
  */
case object WriteDeletedAnnotation extends NoTargetAnnotation with StageOption with HasShellOptions {

  val options = Seq(
    new ShellOption[Unit](
      longOption = "write-deleted",
      toAnnotationSeq = (_: Unit) => Seq(WriteDeletedAnnotation),
      helpText = "Include deleted annotations in the output annotation file" ) )

}
