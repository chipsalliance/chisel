// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

case class StageException(val str: String, cause: Throwable = null) extends RuntimeException(str, cause)

/** A [[Stage]] represents one stage in the FIRRTL hardware compiler framework. A [[Stage]] is, conceptually, a
  * [[Phase]] that includes a command line interface.
  *
  * The FIRRTL compiler is a stage as well as any frontend or backend that runs before/after FIRRTL. Concretely, Chisel
  * is a [[Stage]] as is FIRRTL's Verilog emitter. Each stage performs a mathematical transformation on an
  * [[AnnotationSeq]] where some input annotations are processed to produce different annotations. Command line options
  * may be pulled in if available.
  */
abstract class Stage extends Phase {

  /** A utility that helps convert command line options to annotations */
  val shell: Shell

  /** Run this [[Stage]] on some input annotations
    * @param annotations input annotations
    * @return output annotations
    */
  def run(annotations: AnnotationSeq): AnnotationSeq

  /** Execute this [[Stage]] on some input annotations. Annotations will be read from any input annotation files.
    * @param annotations input annotations
    * @return output annotations
    */
  final def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val preprocessing: Seq[Phase] = Seq(
      phases.GetIncludes,
      phases.ConvertLegacyAnnotations,
      phases.AddDefaults )

    val a = preprocessing.foldLeft(annotations)((a, p) => p.transform(a))

    run(a)
  }

  /** Run this [[Stage]] on on a mix of arguments and annotations
    * @param args command line arguments
    * @param initialAnnotations annotation
    * @return output annotations
    */
  final def execute(args: Array[String], annotations: AnnotationSeq): AnnotationSeq =
    transform(shell.parse(args, annotations))

  /** The main function that serves as this [[Stage]]'s command line interface
    * @param args command line arguments
    */
  final def main(args: Array[String]): Unit = execute(args, Seq.empty)
}
