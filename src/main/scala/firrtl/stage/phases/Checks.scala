// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.stage._

import firrtl.{AnnotationSeq, EmitAllModulesAnnotation, EmitCircuitAnnotation}
import firrtl.annotations.Annotation
import firrtl.options.{Dependency, OptionsException, Phase, PreservesAll}

/** [[firrtl.options.Phase Phase]] that strictly validates an [[AnnotationSeq]]. The checks applied are intended to be
  * extremeley strict. Nothing is inferred or assumed to take a default value (for default value resolution see
  * [[AddDefaults]]).
  *
  * The intent of this approach is that after running this [[firrtl.options.Phase Phase]], a user can be absolutely
  * certain that other [[firrtl.options.Phase Phase]]s or views will succeed. See [[FirrtlStage]] for a list of
  * [[firrtl.options.Phase Phase]] that commonly run before this.
  */
class Checks extends Phase with PreservesAll[Phase] {

  override val prerequisites = Seq(Dependency[AddDefaults], Dependency[AddImplicitEmitter])

  override val dependents = Seq.empty

  /** Determine if annotations are sane
    *
    * @param annos a sequence of [[firrtl.annotations.Annotation Annotation]]
    * @return true if all checks pass
    * @throws firrtl.options.OptionsException if any checks fail
    */
  def transform(annos: AnnotationSeq): AnnotationSeq = {
    val inF, inS, eam, ec, outF, comp, im, inC = collection.mutable.ListBuffer[Annotation]()
    annos.foreach(
      _ match {
        case a: FirrtlFileAnnotation     => a +=: inF
        case a: FirrtlSourceAnnotation   => a +=: inS
        case a: EmitAllModulesAnnotation => a +=: eam
        case a: EmitCircuitAnnotation    => a +=: ec
        case a: OutputFileAnnotation     => a +=: outF
        case a: CompilerAnnotation       => a +=: comp
        case a: InfoModeAnnotation       => a +=: im
        case a: FirrtlCircuitAnnotation  => a +=: inC
        case _                           =>           })

    /* At this point, only a FIRRTL Circuit should exist */
    if (inF.isEmpty && inS.isEmpty && inC.isEmpty) {
      throw new OptionsException(
        s"""|Unable to determine FIRRTL source to read. None of the following were found:
            |    - an input file:  -i, --input-file,    FirrtlFileAnnotation
            |    - FIRRTL source:      --firrtl-source, FirrtlSourceAnnotation
            |    - FIRRTL circuit:                      FirrtlCircuitAnnotation""".stripMargin )}

    /* Only one FIRRTL input can exist */
    if (inF.size + inS.size + inC.size > 1) {
      throw new OptionsException(
        s"""|Multiply defined input FIRRTL sources. More than one of the following was found:
            |    - an input file (${inF.size} times):  -i, --input-file,    FirrtlFileAnnotation
            |    - FIRRTL source (${inS.size} times):      --firrtl-source, FirrtlSourceAnnotation
            |    - FIRRTL circuit (${inC.size} times):                      FirrtlCircuitAnnotation""".stripMargin )}

    /* Specifying an output file and one-file-per module conflict */
    if (eam.nonEmpty && outF.nonEmpty) {
      throw new OptionsException(
        s"""|Output file is incompatible with emit all modules annotation, but multiples were found:
            |    - explicit output file (${outF.size} times): -o, --output-file, OutputFileAnnotation
            |    - one file per module (${eam.size} times):  -e, --emit-modules, EmitAllModulesAnnotation"""
          .stripMargin )}

    /* Only one output file can be specified */
    if (outF.size > 1) {
      val x = outF.map{ case OutputFileAnnotation(x) => x }
      throw new OptionsException(
        s"""|No more than one output file can be specified, but found '${x.mkString(", ")}' specified via:
            |    - option or annotation: -o, --output-file, OutputFileAnnotation""".stripMargin) }

    /* One mandatory compiler must be specified */
    if (comp.size != 1) {
      val x = comp.map{ case CompilerAnnotation(x) => x }
      val (msg, suggest) = if (comp.size == 0) { ("none found",                       "forget one of")   }
      else                                     { (s"""found '${x.mkString(", ")}'""", "use multiple of") }
      throw new OptionsException(
        s"""|Exactly one compiler must be specified, but $msg. Did you $suggest the following?
            |    - an option or annotation: -X, --compiler, CompilerAnnotation""".stripMargin )}

    /* One mandatory info mode must be specified */
    if (im.size != 1) {
      val x = im.map{ case InfoModeAnnotation(x) => x }
      val (msg, suggest) = if (im.size == 0) { ("none found",                       "forget one of")   }
      else                                     { (s"""found '${x.mkString(", ")}'""", "use multiple of") }
      throw new OptionsException(
        s"""|Exactly one info mode must be specified, but $msg. Did you $suggest the following?
            |    - an option or annotation: --info-mode, InfoModeAnnotation""".stripMargin )}

    annos
  }

}
