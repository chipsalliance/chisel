// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.stage.ChiselGeneratorAnnotation

import firrtl.{
  AnnotationSeq,
  EmittedVerilogCircuit,
  EmittedVerilogCircuitAnnotation
}
import firrtl.options.{
  Dependency,
  OptionsException,
  Phase,
  PhaseManager,
  Shell,
  Stage,
  StageError,
  StageMain,
  StageOptions,
  StageUtils
}
import firrtl.options.Viewer.view
import firrtl.stage.{
  FirrtlCircuitAnnotation,
  FirrtlCli,
  FirrtlFileAnnotation,
  FirrtlOptions,
  OutputFileAnnotation
}

import java.io.File

import scala.sys.process._

trait CLI { this: Shell =>
  parser.note("CIRCT (MLIR FIRRTL Compiler) options")
  Seq(
    CIRCTTargetAnnotation,
    DisableLowerTypes,
    ChiselGeneratorAnnotation
  ).foreach(_.addOptions(parser))
}

class CIRCTStage extends Stage {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override val shell: Shell = new Shell("circt") with CLI with FirrtlCli

  final val phaseManager = new PhaseManager(
    targets = Seq(
      Dependency[circt.stage.phases.CIRCT]
    ),
    currentState = Seq(
      Dependency[firrtl.stage.phases.AddImplicitEmitter]
    )
  )

  override def run(annotations: AnnotationSeq): AnnotationSeq = phaseManager.transform(annotations)

}

object CIRCTMain extends StageMain(new CIRCTStage)
