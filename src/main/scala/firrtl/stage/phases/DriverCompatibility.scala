// See LICENSE for license details.

package firrtl.stage.phases

import firrtl.stage._

import firrtl.{AnnotationSeq, EmitAllModulesAnnotation, EmitCircuitAnnotation, FirrtlExecutionResult, Parser}
import firrtl.annotations.NoTargetAnnotation
import firrtl.FileUtils
import firrtl.proto.FromProto
import firrtl.options.{InputAnnotationFileAnnotation, OptionsException, Phase, PreservesAll, StageOptions, StageUtils}
import firrtl.options.Viewer
import firrtl.options.Dependency

import scopt.OptionParser

import java.io.File

/** Provides compatibility methods to replicate deprecated [[Driver]] semantics.
  *
  * At a high level, the [[Driver]] tries extremely hard to figure out what the user meant and to enable them to not be
  * explicit with command line options. As an example, the `--top-name` option is not used for any FIRRTL top module
  * determination, but to find a FIRRTL file by that name and/or an annotation file by that name. This mode of file
  * discovery is only used if no explicit FIRRTL file/source/circuit and/or annotation file is given. Going further, the
  * `--top-name` argument is implicitly specified by the `main` of an input circuit if not explicit and can be used to
  * derive an annotation file. Summarily, the [[firrtl.options.Phase Phase]]s provided by this enable this type of
  * resolution.
  *
  * '''Only use these methods if you are intending to replicate old [[Driver]] semantics for a good reason.'''
  * Otherwise, opt for more explicit specification by the user.
  */
object DriverCompatibility {

  /** Shorthand object for throwing an exception due to an option that was removed */
  private def optionRemoved(a: String): String =
    s"""|Option '$a' was removed as part of the FIRRTL Stage refactor. Use an explicit input/output options instead.
        |This error will be removed in 1.3.""".stripMargin

  /** Convert an [[firrtl.AnnotationSeq AnnotationSeq]] to a ''deprecated'' [[firrtl.FirrtlExecutionResult
    * FirrtlExecutionResult]].
    * @param annotations a sequence of [[firrtl.annotations.Annotation Annotation]]
    */
  @deprecated("FirrtlExecutionResult is deprecated as part of the Stage/Phase refactor. Migrate to FirrtlStage.", "1.2")
  def firrtlResultView(annotations: AnnotationSeq): FirrtlExecutionResult =
    Viewer[FirrtlExecutionResult].view(annotations)

  /** Holds the name of the top (main) module in an input circuit
    * @param value top module name
    */
  @deprecated(""""top-name" is deprecated as part of the Stage/Phase refactor. Use explicit input/output files.""", "1.2")
  case class TopNameAnnotation(topName: String) extends NoTargetAnnotation

  object TopNameAnnotation {

    def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
      .opt[Unit]("top-name")
      .abbr("tn")
      .hidden
      .unbounded
      .action( (_, _) => throw new OptionsException(optionRemoved("--top-name/-tn")) )
  }

  /** Indicates that the implicit emitter, derived from a [[CompilerAnnotation]] should be an [[EmitAllModulesAnnotation]]
    * as opposed to an [[EmitCircuitAnnotation]].
    */
  case object EmitOneFilePerModuleAnnotation extends NoTargetAnnotation {

    def addOptions(p: OptionParser[AnnotationSeq]): Unit = p
      .opt[Unit]("split-modules")
      .abbr("fsm")
      .hidden
      .unbounded
      .action( (_, _) => throw new OptionsException(optionRemoved("--split-modules/-fsm")) )

  }

  /** Determine the top name using the following precedence (highest to lowest):
    *  - Explicitly from a [[TopNameAnnotation]]
    *  - Implicitly from the top module ([[firrtl.ir.Circuit.main main]]) of a [[FirrtlCircuitAnnotation]]
    *  - Implicitly from the top module ([[firrtl.ir.Circuit.main main]]) of a [[FirrtlSourceAnnotation]]
    *  - Implicitly from the top module ([[firrtl.ir.Circuit.main main]]) of a [[FirrtlFileAnnotation]]
    *
    * @param annotations annotations to extract topName from
    * @return the top module ''if it can be determined''
    */
  private def topName(annotations: AnnotationSeq): Option[String] =
    annotations.collectFirst{ case TopNameAnnotation(n) => n }.orElse(
      annotations.collectFirst{ case FirrtlCircuitAnnotation(c) => c.main }.orElse(
        annotations.collectFirst{ case FirrtlSourceAnnotation(s) => Parser.parse(s).main }.orElse(
          annotations.collectFirst{ case FirrtlFileAnnotation(f) =>
            FirrtlStageUtils.getFileExtension(f) match {
              case ProtoBufFile => FromProto.fromFile(f).main
              case FirrtlFile   => Parser.parse(FileUtils.getText(f)).main } } )))

  /** Determine the target directory with the following precedence (highest to lowest):
    *  - Explicitly from the user-specified [[firrtl.options.TargetDirAnnotation TargetDirAnnotation]]
    *  - Implicitly from the default of [[firrtl.options.StageOptions.targetDir StageOptions.targetDir]]
    *
    * @param annotations input annotations to extract targetDir from
    * @return the target directory
    */
  private def targetDir(annotations: AnnotationSeq): String = Viewer[StageOptions].view(annotations).targetDir

  /** Add an implicit annotation file derived from the determined top name of the circuit if no
    * [[firrtl.options.InputAnnotationFileAnnotation InputAnnotationFileAnnotation]] is present.
    *
    * The implicit annotation file is determined through the following complicated semantics:
    *   - If an [[firrtl.options.InputAnnotationFileAnnotation InputAnnotationFileAnnotation]] already exists, then
    *     nothing is modified
    *   - If the derived topName (the `main` in a [[firrtl.ir.Circuit Circuit]]) is ''discernable'' (see below) and a
    *     file called `topName.anno` (exactly, not `topName.anno.json`) exists, then this will add an
    *     [[firrtl.options.InputAnnotationFileAnnotation InputAnnotationFileAnnotation]] using that `topName.anno`
    *   - If any of this doesn't work, then the the [[AnnotationSeq]] is unmodified
    *
    * The precedence for determining the `topName` is the following (first one wins):
    *   - The `topName` in a [[TopNameAnnotation]]
    *   - The `main` [[FirrtlCircuitAnnotation]]
    *   - The `main` in a parsed [[FirrtlSourceAnnotation]]
    *   - The `main` in the first [[FirrtlFileAnnotation]] using either ProtoBuf or parsing as determined by file
    *     extension
    *
    * @param annos input annotations
    * @return output annotations
    */
  class AddImplicitAnnotationFile extends Phase with PreservesAll[Phase] {

    override val prerequisites = Seq(Dependency[AddImplicitFirrtlFile])

    override val dependents = Seq(Dependency[FirrtlPhase], Dependency[FirrtlStage])

    /** Try to add an [[firrtl.options.InputAnnotationFileAnnotation InputAnnotationFileAnnotation]] implicitly specified by
      * an [[AnnotationSeq]]. */
    def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
      .collectFirst{ case a: InputAnnotationFileAnnotation => a } match {
        case Some(_) => annotations
        case None => topName(annotations) match {
          case Some(n) =>
            val filename = targetDir(annotations) + "/" + n + ".anno"
            if (new File(filename).exists) {
              StageUtils.dramaticWarning(
                s"Implicit reading of the annotation file is deprecated! Use an explict --annotation-file argument.")
              annotations :+ InputAnnotationFileAnnotation(filename)
            } else {
              annotations
            }
          case None => annotations
        } }

  }

  /** Add a [[FirrtlFileAnnotation]] if no annotation that explictly defines a circuit exists.
    *
    * This takes the option with the following precedence:
    *  - If an annotation subclassing [[CircuitOption]] exists, do nothing
    *  - If a [[TopNameAnnotation]] exists, use that to derive a [[FirrtlFileAnnotation]] and append it
    *  - Do nothing
    *
    * In the case of (3) above, this [[AnnotationSeq]] is likely insufficient for FIRRTL to work with (no circuit was
    * passed). However, instead of catching this here, we rely on [[Checks]] to validate the annotations.
    *
    * @param annotations input annotations
    * @return
    */
  class AddImplicitFirrtlFile extends Phase with PreservesAll[Phase] {

    override val prerequisites = Seq.empty

    override val dependents = Seq(Dependency[FirrtlPhase], Dependency[FirrtlStage])

    /** Try to add a [[FirrtlFileAnnotation]] implicitly specified by an [[AnnotationSeq]]. */
    def transform(annotations: AnnotationSeq): AnnotationSeq = {
      val circuit = annotations.collectFirst { case a @ (_: CircuitOption | _: FirrtlCircuitAnnotation) => a }
      val main = annotations.collectFirst { case a: TopNameAnnotation => a.topName }

      if (circuit.nonEmpty) {
        annotations
      } else if (main.nonEmpty) {
        StageUtils.dramaticWarning(
          s"Implicit reading of the input file is deprecated! Use an explict --input-file argument.")
        FirrtlFileAnnotation(Viewer[StageOptions].view(annotations).getBuildFileName(s"${main.get}.fir")) +: annotations
      } else {
        annotations
      }
    }
  }

  /** Adds an [[firrtl.EmitAnnotation EmitAnnotation]] for each [[CompilerAnnotation]].
    *
    * If an [[EmitOneFilePerModuleAnnotation]] exists, then this will add an [[EmitAllModulesAnnotation]]. Otherwise,
    * this adds an [[EmitCircuitAnnotation]]. This replicates old behavior where specifying a compiler automatically
    * meant that an emitter would also run.
    */
  @deprecated("""AddImplicitEmitter should only be used to build Driver compatibility wrappers. Switch to Stage.""",
              "1.2")
  class AddImplicitEmitter extends Phase with PreservesAll[Phase] {

    override val prerequisites = Seq.empty

    override val dependents = Seq(Dependency[FirrtlPhase], Dependency[FirrtlStage])

    /** Add one [[EmitAnnotation]] foreach [[CompilerAnnotation]]. */
    def transform(annotations: AnnotationSeq): AnnotationSeq = {
      val splitModules = annotations.collectFirst{ case a: EmitOneFilePerModuleAnnotation.type => a }.isDefined

      annotations.flatMap {
        case a @ CompilerAnnotation(c) =>
          val b = RunFirrtlTransformAnnotation(a.compiler.emitter)
          if (splitModules) { Seq(a, b, EmitAllModulesAnnotation(c.emitter.getClass)) }
          else              { Seq(a, b, EmitCircuitAnnotation   (c.emitter.getClass)) }
        case a => Seq(a)
      }
    }

  }

  /** Adds an [[OutputFileAnnotation]] derived from a [[TopNameAnnotation]] if no [[OutputFileAnnotation]] already
    * exists. If no [[TopNameAnnotation]] exists, then no [[OutputFileAnnotation]] is added.
    */
  @deprecated("""AddImplicitOutputFile should only be used to build Driver compatibility wrappers. Switch to Stage.""",
              "1.2")
  class AddImplicitOutputFile extends Phase with PreservesAll[Phase] {

    override val prerequisites = Seq(Dependency[AddImplicitFirrtlFile])

    override val dependents = Seq(Dependency[FirrtlPhase], Dependency[FirrtlStage])

    /** Add an [[OutputFileAnnotation]] derived from a [[TopNameAnnotation]] if needed. */
    def transform(annotations: AnnotationSeq): AnnotationSeq = {
      val hasOutputFile = annotations
        .collectFirst{ case a @(_: EmitOneFilePerModuleAnnotation.type | _: OutputFileAnnotation) => a }
        .isDefined
      val top = topName(annotations)

      if (!hasOutputFile && top.isDefined) {
        OutputFileAnnotation(top.get) +: annotations
      } else {
        annotations
      }
    }

  }

}
