// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

/** Utilities mixed into something that looks like a [[Stage]] */
object StageUtils {
  /** Print a warning message (in yellow)
    * @param message error message
    */
  //scalastyle:off regex
  def dramaticWarning(message: String): Unit = {
    println(Console.YELLOW + "-"*78)
    println(s"Warning: $message")
    println("-"*78 + Console.RESET)
  }

  /** Print an error message (in red)
    * @param message error message
    * @note This does not stop the Driver.
    */
  //scalastyle:off regex
  def dramaticError(message: String): Unit = {
    println(Console.RED + "-"*78)
    println(s"Error: $message")
    println("-"*78 + Console.RESET)
  }
}

/** A [[Stage]] represents one stage in the FIRRTL hardware compiler framework
  *
  * The FIRRTL compiler is a stage as well as any frontend or backend that runs before/after FIRRTL. Concretely, Chisel
  * is a [[Stage]] as is FIRRTL's Verilog emitter. Each stage performs a mathematical transformation on an
  * [[AnnotationSeq]] where some input annotations are processed to produce different annotations. Command line options
  * may be pulled in if available.
  */
abstract class Stage {

  /** A utility that helps convert command line options to annotations */
  val shell: Shell

  /** Run this [[Stage]] on some input annotations
    * @param annotations input annotations
    * @return output annotations
    */
  def execute(annotations: AnnotationSeq): AnnotationSeq

  /** Run this [[Stage]] on on a mix of arguments and annotations
    * @param args command line arguments
    * @param initialAnnotations annotation
    * @return output annotations
    */
  def execute(args: Array[String], annotations: AnnotationSeq): AnnotationSeq =
    execute(shell.parse(args, annotations))

  /** The main function that serves as this [[Stage]]'s command line interface
    * @param args command line arguments
    */
  def main(args: Array[String]): Unit = execute(args, Seq.empty)
}
