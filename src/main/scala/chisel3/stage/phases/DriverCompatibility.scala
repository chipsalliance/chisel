// See LICENSE for license details.

package chisel3.stage.phases

import firrtl.{AnnotationSeq, ExecutionOptionsManager, HasFirrtlOptions}
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.{Dependency, OptionsException, OutputAnnotationFileAnnotation, Phase, Unserializable}
import firrtl.stage.{FirrtlCircuitAnnotation, RunFirrtlTransformAnnotation}
import firrtl.stage.phases.DriverCompatibility.TopNameAnnotation

import chisel3.HasChiselExecutionOptions
import chisel3.stage.{ChiselStage, NoRunFirrtlCompilerAnnotation, ChiselOutputFileAnnotation}

/** This provides components of a compatibility wrapper around Chisel's deprecated [[chisel3.Driver]].
  *
  * Primarily, this object includes [[firrtl.options.Phase Phase]]s that generate [[firrtl.annotations.Annotation]]s
  * derived from the deprecated [[firrtl.stage.phases.DriverCompatibility.TopNameAnnotation]].
  */
object DriverCompatibility {

  /** Adds a [[ChiselOutputFileAnnotation]] derived from a [[TopNameAnnotation]] if no [[ChiselOutputFileAnnotation]]
    * already exists. If no [[TopNameAnnotation]] exists, then no [[firrtl.stage.OutputFileAnnotation]] is added. ''This is not a
    * replacement for [[chisel3.stage.phases.AddImplicitOutputFile AddImplicitOutputFile]] as this only adds an output
    * file based on a discovered top name and not on a discovered elaborated circuit.'' Consequently, this will provide
    * the correct behavior before a circuit has been elaborated.
    * @note the output suffix is unspecified and will be set by the underlying [[firrtl.EmittedComponent]]
    */
  private [chisel3] class AddImplicitOutputFile extends Phase {

    override def prerequisites = Seq.empty
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[chisel3.stage.ChiselStage])
    override def invalidates(a: Phase) = false

    def transform(annotations: AnnotationSeq): AnnotationSeq = {
      val hasOutputFile = annotations
        .collectFirst{ case a: ChiselOutputFileAnnotation => a }
        .isDefined
      lazy val top = annotations.collectFirst{ case TopNameAnnotation(a) => a }

      if (!hasOutputFile && top.isDefined) {
        ChiselOutputFileAnnotation(top.get) +: annotations
      } else {
        annotations
      }
    }
  }

  /** If a [[firrtl.options.OutputAnnotationFileAnnotation]] does not exist, this adds one derived from a
    * [[TopNameAnnotation]]. ''This is not a replacement for [[chisel3.stage.phases.AddImplicitOutputAnnotationFile]] as
    * this only adds an output annotation file based on a discovered top name.'' Consequently, this will provide the
    * correct behavior before a circuit has been elaborated.
    * @note the output suffix is unspecified and will be set by [[firrtl.options.phases.WriteOutputAnnotations]]
    */
  private[chisel3] class AddImplicitOutputAnnotationFile extends Phase {

    override def prerequisites = Seq.empty
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[chisel3.stage.ChiselStage])
    override def invalidates(a: Phase) = false

    def transform(annotations: AnnotationSeq): AnnotationSeq =
      annotations
        .collectFirst{ case _: OutputAnnotationFileAnnotation => annotations }
        .getOrElse{
          val top = annotations.collectFirst{ case TopNameAnnotation(a) => a }
          if (top.isDefined) {
            OutputAnnotationFileAnnotation(top.get) +: annotations
          } else {
            annotations
          }
      }
  }

  private[chisel3] case object RunFirrtlCompilerAnnotation extends NoTargetAnnotation

  /** Disables the execution of [[firrtl.stage.FirrtlStage]]. This can be used to call [[chisel3.stage.ChiselStage]] and
    * guarantee that the FIRRTL compiler will not run. This is necessary for certain [[chisel3.Driver]] compatibility
    * situations where you need to do something between Chisel compilation and FIRRTL compilations, e.g., update a
    * mutable data structure.
    */
  private[chisel3] class DisableFirrtlStage extends Phase {

    override def prerequisites = Seq.empty
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[ChiselStage])
    override def invalidates(a: Phase) = false

    def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
      .collectFirst { case NoRunFirrtlCompilerAnnotation => annotations                              }
      .getOrElse    { Seq(RunFirrtlCompilerAnnotation, NoRunFirrtlCompilerAnnotation) ++ annotations }
  }

  private[chisel3] class ReEnableFirrtlStage extends Phase {

    override def prerequisites = Seq(Dependency[DisableFirrtlStage], Dependency[ChiselStage])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq.empty
    override def invalidates(a: Phase) = false

    def transform(annotations: AnnotationSeq): AnnotationSeq = annotations
      .collectFirst { case RunFirrtlCompilerAnnotation =>
        val a: AnnotationSeq = annotations.filter {
          case NoRunFirrtlCompilerAnnotation | RunFirrtlCompilerAnnotation => false
          case _                                                           => true
        }
        a
      }
      .getOrElse{ annotations }

  }

  private[chisel3] case class OptionsManagerAnnotation(
    manager: ExecutionOptionsManager with HasChiselExecutionOptions with HasFirrtlOptions)
      extends NoTargetAnnotation with Unserializable

  /** Mutate an input [[firrtl.ExecutionOptionsManager]] based on information encoded in an [[firrtl.AnnotationSeq]].
    * This is intended to be run between [[chisel3.stage.ChiselStage ChiselStage]] and [[firrtl.stage.FirrtlStage]] if
    * you want to have backwards compatibility with an [[firrtl.ExecutionOptionsManager]].
    */
  private[chisel3] class MutateOptionsManager extends Phase {

    override def prerequisites = Seq(Dependency[chisel3.stage.ChiselStage])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[ReEnableFirrtlStage])
    override def invalidates(a: Phase) = false

    def transform(annotations: AnnotationSeq): AnnotationSeq = {

      val optionsManager = annotations
        .collectFirst{ case OptionsManagerAnnotation(a) => a }
        .getOrElse{ throw new OptionsException(
                     "An OptionsManagerException must exist for Chisel Driver compatibility mode") }

      val firrtlCircuit = annotations.collectFirst{ case FirrtlCircuitAnnotation(a) => a }
      optionsManager.firrtlOptions = optionsManager.firrtlOptions.copy(
        firrtlCircuit = firrtlCircuit,
        annotations = optionsManager.firrtlOptions.annotations ++ annotations,
        customTransforms = optionsManager.firrtlOptions.customTransforms ++
          annotations.collect{ case RunFirrtlTransformAnnotation(a) => a } )

      annotations

    }

  }

  /** A [[Phase]] that lets us run
    * @todo a better solution than the current state hack below may be needed
    */
  private [chisel3] class FirrtlPreprocessing extends Phase {

    override def prerequisites = Seq(Dependency[ChiselStage], Dependency[MutateOptionsManager], Dependency[ReEnableFirrtlStage])
    override def optionalPrerequisites = Seq.empty
    override def optionalPrerequisiteOf = Seq(Dependency[MaybeFirrtlStage])
    override def invalidates(a: Phase) = false

    private val phases =
      Seq( new firrtl.stage.phases.DriverCompatibility.AddImplicitOutputFile,
           new firrtl.stage.phases.DriverCompatibility.AddImplicitEmitter )

    override def transform(annotations: AnnotationSeq): AnnotationSeq =
      phases
        .foldLeft(annotations)( (a, p) => p.transform(a) )

  }

}
