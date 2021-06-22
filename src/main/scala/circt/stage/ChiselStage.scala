// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.RawModule
import chisel3.stage.{
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
import firrtl.options.Viewer.view
import firrtl.stage.{
  Forms,
  RunFirrtlTransformAnnotation
}

/** Entry point for running Chisel with the CIRCT compiler.
  *
  * This is intended to be a replacement for [[chisel3.stage.ChiselStage]].
  *
  * @note The companion object, [[ChiselStage$]], has a cleaner API for compiling and returning a string.
  */
class ChiselStage extends Stage {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override val shell = new Shell("circt") with CLI

  override def run(annotations: AnnotationSeq): AnnotationSeq = {

    val pm = new PhaseManager(
      targets = Seq(
        Dependency[chisel3.stage.ChiselStage],
        Dependency[firrtl.stage.phases.AddImplicitOutputFile],
        Dependency[circt.stage.phases.AddDefaults],
        Dependency[circt.stage.phases.Checks],
        Dependency[circt.stage.phases.MaybeSFC],
        Dependency[circt.stage.CIRCTStage]
      ),
      currentState = Seq(
        Dependency[firrtl.stage.phases.AddDefaults],
        Dependency[firrtl.stage.phases.Checks]
      )
    )
    pm.transform(NoRunFirrtlCompilerAnnotation +: annotations)
  }

}

/** Utilities for compiling Chisel */
object ChiselStage {

  /** Elaborate a Chisel circuit into a CHIRRTL string */
  def emitCHIRRTL(gen: => RawModule): String = chisel3.stage.ChiselStage.emitChirrtl(gen)

  /** A phase shared by all the CIRCT backends */
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

  /** Compile a Chisel circuit to FIRRTL dialect */
  def emitFIRRTLDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.FIRRTL),
        CIRCTHandover(CIRCTHandover.CHIRRTL)
      )
    ).collectFirst {
      case EmittedMLIR(_, a, _) => a
    }.get

  /** Compile a Chisel circuit to HWS dialect */
  def emitHWDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.HW),
        CIRCTHandover(CIRCTHandover.CHIRRTL)
      )
    ).collectFirst {
      case EmittedMLIR(_, a, _) => a
    }.get

  /** Compile a Chisel circuit to SystemVerilog */
  def emitSystemVerilog(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog),
        CIRCTHandover(CIRCTHandover.CHIRRTL)
      )
    ).collectFirst {
      case EmittedVerilogCircuitAnnotation(a) => a
    }.get
    .value

}

/** Command line entry point to [[ChiselStage]] */
object ChiselMain extends StageMain(new ChiselStage)
