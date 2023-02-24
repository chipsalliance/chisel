// SPDX-License-Identifier: Apache-2.0

package firrtl.options

import firrtl.AnnotationSeq
import firrtl.annotations.{Annotation, NoTargetAnnotation}
import firrtl.options.Viewer.view

import java.io.File

import scopt.OptionParser

sealed trait StageOption extends Unserializable { this: Annotation => }

/** An annotation that should not be serialized automatically [[phases.WriteOutputAnnotations WriteOutputAnnotations]].
  * This usually means that this is an annotation that is used only internally to a [[Stage]].
  */
trait Unserializable { this: Annotation => }

/** Mix-in that lets an [[firrtl.annotations.Annotation Annotation]] serialize itself to a file separate from the output
  * annotation file.
  *
  * This can be used as a mechanism to serialize an [[firrtl.options.Unserializable Unserializable]] annotation or to
  * write ancillary collateral used by downstream tooling, e.g., a TCL script or an FPGA constraints file. Any
  * annotations using this mix-in will be serialized by the [[firrtl.options.phases.WriteOutputAnnotations
  * WriteOutputAnnotations]] phase. This is one of the last phases common to all [[firrtl.options.Stage Stages]] and
  * should not have to be called/included manually.
  *
  * Note: from the perspective of transforms generating annotations that mix-in this trait, the serialized files are not
  * expected to be available to downstream transforms. Communication of information between transforms must occur
  * through the annotations that will eventually be serialized to files.
  */
trait CustomFileEmission { this: Annotation =>

  /** Output filename where serialized content will be written
    *
    * The full annotation sequence is a parameter to allow for the location where this annotation will be serialized to
    * be a function of other annotations, e.g., if the location where information is written is controlled by a separate
    * file location annotation.
    *
    * @param annotations the annotation sequence at the time of emission
    */
  protected def baseFileName(annotations: AnnotationSeq): String

  /** Optional suffix of the output file */
  protected def suffix: Option[String]

  /** A method that can convert this annotation to bytes that will be written to a file.
    *
    * If you only need to serialize a string, you can use the `getBytes` method:
    * {{{
    *  def getBytes: Iterable[Byte] = myString.getBytes
    * }}}
    */
  def getBytes: Iterable[Byte]

  /** Optionally, a sequence of annotations that will replace this annotation in the output annotation file.
    *
    * A non-empty implementation of this method is a mechanism for telling a downstream [[firrtl.options.Stage Stage]]
    * how to deserialize the information that was serialized to a separate file. For example, if a FIRRTL circuit is
    * serialized to a separate file, this method could include an input file annotation that a later stage can use to
    * read the serialized FIRRTL circuit back in.
    */
  def replacements(file: File): AnnotationSeq = Seq.empty

  /** Method that returns the filename where this annotation will be serialized.
    *
    * @param annotations the annotations at the time of serialization
    */
  final def filename(annotations: AnnotationSeq): File = {
    val name = view[StageOptions](annotations).getBuildFileName(baseFileName(annotations), suffix)
    new File(name)
  }
}

/** A buffered version of [[CustomFileEmission]]
  *
  * This is especially useful for serializing large data structures. When emitting Strings, it makes
  * it much easier to avoid materializing the entire serialized String in memory. It also helps
  * avoid materializing intermediate datastructures in memory. Finally, it reduces iteration
  * overhead and helps optimize I/O performance.
  *
  * It may seem strange to use `Array[Byte]` in an otherwise immutable API, but for maximal
  * performance it is best to use the JVM primitive that file I/O uses. These Arrays should only
  * used immutably even though the Java API technically does allow mutating them.
  */
trait BufferedCustomFileEmission extends CustomFileEmission { this: Annotation =>

  /** A buffered version of [[getBytes]] for more efficient serialization
    *
    * If you only need to serialize an `Iterable[String]`, you can use the `String.getBytes` method.
    * It's also helpful to create a `view` which will do the `.map` lazily instead of eagerly,
    * improving GC performance.
    * {{{
    *  def getBytesBuffered: Iterable[Array[Byte]] = myListString.view.map(_.getBytes)
    * }}}
    */
  def getBytesBuffered: Iterable[Array[Byte]]

  final def getBytes: Iterable[Byte] = getBytesBuffered.flatten
}

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
      helpValueName = Some("<directory>")
    )
  )

}

/** Additional arguments
  *  - set with any trailing option on the command line
  * @param value one [[scala.Predef.String String]] argument
  */
case class ProgramArgsAnnotation(arg: String) extends NoTargetAnnotation with StageOption

object ProgramArgsAnnotation {

  def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
    .arg[String]("<arg>...")
    .unbounded()
    .optional()
    .action((x, c) => ProgramArgsAnnotation(x) +: c)
    .text("optional unbounded args")
}

/** Holds a filename containing one or more [[annotations.Annotation]] to be read
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
      helpValueName = Some("<file>")
    )
  )

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
      helpValueName = Some("<file>")
    )
  )

}
