// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.deprecatedMFCMessage
import chisel3.stage.{
  ChiselGeneratorAnnotation,
  PrintFullStackTraceAnnotation,
  ThrowOnFirstErrorAnnotation,
  WarningsAsErrorsAnnotation
}

import firrtl.AnnotationSeq
import firrtl.options.{Dependency, Phase, PhaseManager, Shell, Stage, StageMain}
import firrtl.stage.FirrtlCli

trait CLI { this: Shell =>
  parser.note("CIRCT (MLIR FIRRTL Compiler) options")
  Seq(
    CIRCTTargetAnnotation,
    PreserveAggregate,
    ChiselGeneratorAnnotation,
    PrintFullStackTraceAnnotation,
    ThrowOnFirstErrorAnnotation,
    WarningsAsErrorsAnnotation,
    SplitVerilog
  ).foreach(_.addOptions(parser))
}

/** A [[firrtl.options.Stage Stage]] used to compile FIRRTL IR using CIRCT. This is a drop-in replacement for
  * [[firrtl.stage.FirrtlStage]].
  *
  * @see [[https://github.com/llvm/circt llvm/circt]]
  */
@deprecated(deprecatedMFCMessage + " Please use circt.stage.Chiselstage.", "Chisel 3.6")
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

/** Command line utility for [[CIRCTStage]]. */
@deprecated(deprecatedMFCMessage + " Please use circt.stage.Chiselstage.", "Chisel 3.6")
object CIRCTMain extends StageMain(new CIRCTStage)
