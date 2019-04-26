// See LICENSE for license details.

package firrtl.options

import firrtl.AnnotationSeq

import logger.Logger

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

  /** Run this stage on some input annotations
    * @param annotations input annotations
    * @return output annotations
    */
  def run(annotations: AnnotationSeq): AnnotationSeq

  /** Execute this stage on some input annotations. Annotations will be read from any input annotation files.
    * @param annotations input annotations
    * @return output annotations
    * @throws OptionsException if command line or annotation validation fails
    */
  final def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val annotationsx =
      Seq( new phases.GetIncludes,
           new phases.ConvertLegacyAnnotations )
        .map(phases.DeletedWrapper(_))
        .foldLeft(annotations)((a, p) => p.transform(a))

    Logger.makeScope(annotationsx) {
      Seq( new phases.AddDefaults,
           new phases.Checks,
           new Phase { def transform(a: AnnotationSeq) = run(a) },
           new phases.WriteOutputAnnotations )
        .map(phases.DeletedWrapper(_))
        .foldLeft(annotationsx)((a, p) => p.transform(a))
    }
  }

  /** Run this stage on on a mix of arguments and annotations
    * @param args command line arguments
    * @param initialAnnotations annotation
    * @return output annotations
    * @throws OptionsException if command line or annotation validation fails
    */
  final def execute(args: Array[String], annotations: AnnotationSeq): AnnotationSeq =
    transform(shell.parse(args, annotations))

}

/** Provides a main method for a [[Stage]]
  * @param stage the stage to run
  */
class StageMain(val stage: Stage) {

  /** The main function that serves as this stage's command line interface.
    * @param args command line arguments
    */
  final def main(args: Array[String]): Unit = try {
    stage.execute(args, Seq.empty)
  } catch {
    case a: OptionsException =>
      StageUtils.dramaticUsageError(a.message)
      System.exit(1)
  }

}
