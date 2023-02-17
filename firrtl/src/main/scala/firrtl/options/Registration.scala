// SPDX-License-Identifier: Apache-2.0

package firrtl.options

import firrtl.AnnotationSeq

import scopt.{OptionDef, OptionParser, Read}

/** Contains information about a [[Shell]] command line option
  * @tparam the type of the command line argument
  * @param longOption a long, double-dash option
  * @param toAnnotationSeq a function to convert the type into an [[firrtl.AnnotationSeq AnnotationSeq]]
  * @param helpText help text
  * @param shortOption an optional single-dash option
  * @param helpValueName a string to show as a placeholder argument in help text
  */
final class ShellOption[A: Read](
  val longOption:      String,
  val toAnnotationSeq: A => AnnotationSeq,
  val helpText:        String,
  val shortOption:     Option[String] = None,
  val helpValueName:   Option[String] = None) {

  /** Add this specific shell (command line) option to an option parser
    * @param p an option parser
    */
  final def addOption(p: OptionParser[AnnotationSeq]): Unit = {
    val f = Seq(
      (p: OptionDef[A, AnnotationSeq]) => p.action((x, c) => toAnnotationSeq(x).reverse ++ c),
      (p: OptionDef[A, AnnotationSeq]) => p.text(helpText),
      (p: OptionDef[A, AnnotationSeq]) => p.unbounded()
    ) ++
      shortOption.map(a => (p: OptionDef[A, AnnotationSeq]) => p.abbr(a)) ++
      helpValueName.map(a => (p: OptionDef[A, AnnotationSeq]) => p.valueName(a))

    f.foldLeft(p.opt[A](longOption))((a, b) => b(a))
  }
}

/** Indicates that this class/object includes options (but does not add these as a registered class)
  */
trait HasShellOptions {

  /** A sequence of options provided
    */
  def options: Seq[ShellOption[_]]

  /** Add all shell (command line) options to an option parser
    * @param p an option parser
    */
  final def addOptions(p: OptionParser[AnnotationSeq]): Unit = options.foreach(_.addOption(p))

}

/** A class that includes options that should be exposed as a group at the top level.
  *
  * @note To complete registration, include an entry in
  * src/main/resources/META-INF/services/firrtl.options.RegisteredLibrary
  */
trait RegisteredLibrary extends HasShellOptions {

  /** The name of this library.
    *
    * This will be used when generating help text.
    */
  def name: String

}
