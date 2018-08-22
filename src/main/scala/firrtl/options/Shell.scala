// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

import scopt.OptionParser

import java.util.ServiceLoader

/** Indicate an error in [[firrtl.options]]
  * @param msg a message to print
  */
case class OptionsException(msg: String, cause: Throwable = null) extends Exception(msg, cause)

/** A utility for working with command line options
  * @param applicationName the application associated with these command line options
  */
class Shell(val applicationName: String) {

  /** Command line argument parser (OptionParser) with modifications */
  final val parser = new OptionParser[AnnotationSeq](applicationName) with DoNotTerminateOnExit with DuplicateHandling

  /** Contains all discovered [[RegisteredLibrary]] */
  lazy val registeredLibraries: Seq[RegisteredLibrary] = {
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
  lazy val registeredTransforms: Seq[RegisteredTransform] = {
    val transforms = scala.collection.mutable.ArrayBuffer[RegisteredTransform]()
    val iter = ServiceLoader.load(classOf[RegisteredTransform]).iterator()
    parser.note("FIRRTL Transform Options")
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
      .parse(args, initAnnos)
      .getOrElse(throw new OptionsException("Failed to parse command line options", new IllegalArgumentException))
  }

}
