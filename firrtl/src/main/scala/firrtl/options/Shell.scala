// SPDX-License-Identifier: Apache-2.0

package firrtl.options

import firrtl.AnnotationSeq

import logger.{ClassLogLevelAnnotation, LogClassNamesAnnotation, LogFileAnnotation, LogLevelAnnotation}

import scopt.OptionParser

import java.util.ServiceLoader

/** A utility for working with command line options.  This provides no options by default other than "--help".  This is
  * intended for lower-level APIs which do not want to include options that are provided by [[Shell]].
  *
  * @param applicationName
  */
class BareShell(val applicationName: String) {

  /** Command line argument parser (OptionParser) with modifications */
  protected val parser = new OptionParser[AnnotationSeq](applicationName) with DuplicateHandling with ExceptOnError
  parser.help("help").text("prints this usage text")

  /** This method can be overriden to do some work everytime before parsing runs, e.g., to add options to the parser. */
  protected def parserSetup(): Unit = {}

  /** The [[AnnotationSeq]] generated from command line arguments
    *
    * This requires lazy evaluation as subclasses will mixin new command
    * line options via methods of [[Shell.parser]]
    */
  def parse(args: Array[String], initAnnos: AnnotationSeq = Seq.empty): AnnotationSeq = {
    parserSetup()
    parser
      .parse(args, initAnnos.reverse)
      .getOrElse(throw new OptionsException("Failed to parse command line options", new IllegalArgumentException))
      .reverse
  }

}

/** A utility for working with command line options.  This comes prepopulated with common options for most uses.
  *
  * @param applicationName the application associated with these command line options
  */
class Shell(applicationName: String) extends BareShell(applicationName) {

  /** Contains all discovered [[RegisteredLibrary]] */
  final lazy val registeredLibraries: Seq[RegisteredLibrary] = {
    val libraries = scala.collection.mutable.ArrayBuffer[RegisteredLibrary]()
    val iter = ServiceLoader.load(classOf[RegisteredLibrary]).iterator()
    while (iter.hasNext) {
      val lib = iter.next()
      libraries += lib
      parser.note(lib.name)
      lib.addOptions(parser)
    }

    libraries.toSeq
  }

  override protected def parserSetup(): Unit = {
    if (sys.env.get("CHISEL_ARGUMENT_EXTENSIONS") != Some("DISABLE")) {
      registeredLibraries
    }
  }

  parser.note("Shell Options")
  ProgramArgsAnnotation.addOptions(parser)
  Seq(TargetDirAnnotation, InputAnnotationFileAnnotation, OutputAnnotationFileAnnotation)
    .foreach(_.addOptions(parser))

  parser
    .opt[Unit]("show-registrations")
    .action { (_, c) =>
      val rlString = registeredLibraries.map(l => s"\n  - ${l.getClass.getName}").mkString

      println(s"""The following libraries registered command line options:$rlString""")
      c
    }
    .unbounded()
    .text("print discovered registered libraries and transforms")

  parser.note("Logging Options")
  Seq(LogLevelAnnotation, ClassLogLevelAnnotation, LogFileAnnotation, LogClassNamesAnnotation)
    .foreach(_.addOptions(parser))
}
