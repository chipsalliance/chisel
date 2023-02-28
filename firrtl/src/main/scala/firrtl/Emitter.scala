// SPDX-License-Identifier: Apache-2.0

package firrtl

import java.io.File
import firrtl.annotations.NoTargetAnnotation
import firrtl.options.Viewer.view
import firrtl.options.{CustomFileEmission, Dependency, HasShellOptions, PhaseException, ShellOption}
import firrtl.passes.PassException
import firrtl.stage.{FirrtlOptions, RunFirrtlTransformAnnotation}

case class EmitterException(message: String) extends PassException(message)

// ***** Annotations for telling the Emitters what to emit *****
sealed trait EmitAnnotation extends NoTargetAnnotation {
  val emitter: Class[_ <: Emitter]
}

case class EmitCircuitAnnotation(emitter: Class[_ <: Emitter]) extends EmitAnnotation

case class EmitAllModulesAnnotation(emitter: Class[_ <: Emitter]) extends EmitAnnotation

object EmitCircuitAnnotation extends HasShellOptions {
  val options = Seq(
    new ShellOption[String](
      longOption = "emit-circuit",
      toAnnotationSeq = (a: String) =>
        a match {
          case "chirrtl" =>
            Seq(RunFirrtlTransformAnnotation(new ChirrtlEmitter), EmitCircuitAnnotation(classOf[ChirrtlEmitter]))
          case "mhigh" =>
            Seq(
              RunFirrtlTransformAnnotation(new MinimumHighFirrtlEmitter),
              EmitCircuitAnnotation(classOf[MinimumHighFirrtlEmitter])
            )
          case "high" =>
            Seq(RunFirrtlTransformAnnotation(new HighFirrtlEmitter), EmitCircuitAnnotation(classOf[HighFirrtlEmitter]))
          case "middle" =>
            Seq(
              RunFirrtlTransformAnnotation(new MiddleFirrtlEmitter),
              EmitCircuitAnnotation(classOf[MiddleFirrtlEmitter])
            )
          case "low" =>
            Seq(RunFirrtlTransformAnnotation(new LowFirrtlEmitter), EmitCircuitAnnotation(classOf[LowFirrtlEmitter]))
          case "low-opt" =>
            Seq(
              RunFirrtlTransformAnnotation(Dependency(LowFirrtlOptimizedEmitter)),
              EmitCircuitAnnotation(LowFirrtlOptimizedEmitter.getClass)
            )
          case _ => throw new PhaseException(s"Unknown emitter '$a'! (Did you misspell it?)")
        },
      helpText = "Run the specified circuit emitter (all modules in one file)",
      shortOption = Some("E"),
      // the experimental options are intentionally excluded from the help message
      helpValueName = Some("<chirrtl|high|middle|low>")
    )
  )
}

object EmitAllModulesAnnotation extends HasShellOptions {

  val options = Seq(
    new ShellOption[String](
      longOption = "emit-modules",
      toAnnotationSeq = (a: String) =>
        a match {
          case "chirrtl" =>
            Seq(RunFirrtlTransformAnnotation(new ChirrtlEmitter), EmitAllModulesAnnotation(classOf[ChirrtlEmitter]))
          case "mhigh" =>
            Seq(
              RunFirrtlTransformAnnotation(new MinimumHighFirrtlEmitter),
              EmitAllModulesAnnotation(classOf[MinimumHighFirrtlEmitter])
            )
          case "high" =>
            Seq(
              RunFirrtlTransformAnnotation(new HighFirrtlEmitter),
              EmitAllModulesAnnotation(classOf[HighFirrtlEmitter])
            )
          case "middle" =>
            Seq(
              RunFirrtlTransformAnnotation(new MiddleFirrtlEmitter),
              EmitAllModulesAnnotation(classOf[MiddleFirrtlEmitter])
            )
          case "low" =>
            Seq(RunFirrtlTransformAnnotation(new LowFirrtlEmitter), EmitAllModulesAnnotation(classOf[LowFirrtlEmitter]))
          case _ => throw new PhaseException(s"Unknown emitter '$a'! (Did you misspell it?)")
        },
      helpText = "Run the specified module emitter (one file per module)",
      shortOption = Some("e"),
      helpValueName = Some("<chirrtl|high|middle|low>")
    )
  )

}

// ***** Annotations for results of emission *****
sealed abstract class EmittedComponent {
  def name: String

  def value: String

  def outputSuffix: String
}

sealed abstract class EmittedCircuit extends EmittedComponent

sealed abstract class EmittedModule extends EmittedComponent

/** Traits for Annotations containing emitted components */
trait EmittedAnnotation[T <: EmittedComponent] extends NoTargetAnnotation with CustomFileEmission {
  val value: T

  override protected def baseFileName(annotations: AnnotationSeq): String = {
    view[FirrtlOptions](annotations).outputFileName.getOrElse(value.name)
  }

  override protected val suffix: Option[String] = Some(value.outputSuffix)
}

sealed trait EmittedCircuitAnnotation[T <: EmittedCircuit] extends EmittedAnnotation[T] {
  override def getBytes = value.value.getBytes
}

sealed trait EmittedModuleAnnotation[T <: EmittedModule] extends EmittedAnnotation[T] {
  override def getBytes = value.value.getBytes
}

case class EmittedFirrtlModuleAnnotation(value: EmittedFirrtlModule)
    extends EmittedModuleAnnotation[EmittedFirrtlModule]
case class EmittedFirrtlCircuitAnnotation(value: EmittedFirrtlCircuit)
    extends EmittedCircuitAnnotation[EmittedFirrtlCircuit]

final case class EmittedFirrtlCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
final case class EmittedFirrtlModule(name: String, value: String, outputSuffix: String) extends EmittedModule

final case class EmittedVerilogCircuit(name: String, value: String, outputSuffix: String) extends EmittedCircuit
final case class EmittedVerilogModule(name: String, value: String, outputSuffix: String) extends EmittedModule
case class EmittedVerilogCircuitAnnotation(value: EmittedVerilogCircuit)
    extends EmittedCircuitAnnotation[EmittedVerilogCircuit]
case class EmittedVerilogModuleAnnotation(value: EmittedVerilogModule)
    extends EmittedModuleAnnotation[EmittedVerilogModule]
