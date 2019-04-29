// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq
import firrtl.transforms.NoCircuitDedupAnnotation

import logger.{LogLevelAnnotation, ClassLogLevelAnnotation, LogFileAnnotation, LogClassNamesAnnotation}

import scopt.OptionParser

import java.util.ServiceLoader

/** A utility for working with command line options
  * @param applicationName the application associated with these command line options
  */
class Shell(val applicationName: String) {

  /** Command line argument parser (OptionParser) with modifications */
  protected val parser = new OptionParser[AnnotationSeq](applicationName) with DuplicateHandling with ExceptOnError

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

    libraries
  }

  /** Contains all discovered [[RegisteredTransform]] */
  final lazy val registeredTransforms: Seq[RegisteredTransform] = {
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredTransform]()
    val iter = ServiceLoader.load(classOf[RegisteredTransform]).iterator()
    if (iter.hasNext) { parser.note("FIRRTL Transform Options") }
    while (iter.hasNext) {
      val tx = iter.next()
      transforms += tx
      tx.addOptions(parser)
    }

    transforms
  }

  /** The [[AnnotationSeq]] generated from command line arguments
    *
    * This requires lazy evaluation as subclasses will mixin new command
    * line options via methods of [[Shell.parser]]
    */
  def parse(args: Array[String], initAnnos: AnnotationSeq = Seq.empty): AnnotationSeq = {
    registeredTransforms
    registeredLibraries
    parser
      .parse(args, initAnnos.reverse)
      .getOrElse(throw new OptionsException("Failed to parse command line options", new IllegalArgumentException))
      .reverse
  }

  parser.note("Shell Options")
  ProgramArgsAnnotation.addOptions(parser)
  Seq( TargetDirAnnotation,
       InputAnnotationFileAnnotation,
       OutputAnnotationFileAnnotation,
       NoCircuitDedupAnnotation)
    .foreach(_.addOptions(parser))

  parser.opt[Unit]("show-registrations")
    .action{ (_, c) =>
      val rtString = registeredTransforms.map(r => s"\n  - ${r.getClass.getName}").mkString
      val rlString = registeredLibraries.map(l => s"\n  - ${l.getClass.getName}").mkString

      println(s"""|The following FIRRTL transforms registered command line options:$rtString
                  |The following libraries registered command line options:$rlString""".stripMargin)
      c }
    .unbounded()
    .text("print discovered registered libraries and transforms")

  parser.help("help").text("prints this usage text")

  parser.note("Logging Options")
  Seq( LogLevelAnnotation,
       ClassLogLevelAnnotation,
       LogFileAnnotation,
       LogClassNamesAnnotation )
    .foreach(_.addOptions(parser))
}
