// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.RawModule
import chisel3.stage.{
  ChiselCli,
  ChiselGeneratorAnnotation,
  NoRunFirrtlCompilerAnnotation
}

import firrtl.{
  AnnotationSeq,
  EmittedVerilogCircuitAnnotation
}
import firrtl.options.{
  Dependency,
  Phase,
  PhaseManager,
  Shell,
  Stage,
  StageMain
}
import firrtl.stage.FirrtlCli

class ChiselStage extends Stage {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override val shell = new Shell("circt") with CLI

  override def run(annotations: AnnotationSeq): AnnotationSeq = {
    val pm = new PhaseManager(
      Seq(
        Dependency[chisel3.stage.ChiselStage],
        Dependency[circt.stage.CIRCTStage]
      )
    )

    pm.transform(NoRunFirrtlCompilerAnnotation +: annotations)
  }

}

object ChiselStage {

  def emitCHIRRTL(gen: => RawModule): String = chisel3.stage.ChiselStage.emitChirrtl(gen)

  private def phase = new PhaseManager(
    Seq(
      Dependency[chisel3.stage.phases.Checks],
      Dependency[chisel3.stage.phases.Elaborate],
      Dependency[chisel3.stage.phases.Convert],
      Dependency[firrtl.stage.phases.AddImplicitOutputFile],
      Dependency[circt.stage.phases.Checks],
      Dependency[circt.stage.phases.CIRCT]
    )
  )

  def emitFIRRTLDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.FIRRTL)
      )
    ).collectFirst {
      case EmittedMLIR(_, a, _) => a
    }.get

  def emitRTLDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.RTL)
      )
    ).collectFirst {
      case EmittedMLIR(_, a, _) => a
    }.get

  def emitSystemVerilog(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog)
      )
    ).collectFirst {
      case EmittedVerilogCircuitAnnotation(a) => a
    }.get
    .value

}

object ChiselMain extends StageMain(new ChiselStage)
