// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.RawModule
import chisel3.stage.{ChiselGeneratorAnnotation, NoRunFirrtlCompilerAnnotation}
import firrtl.options.{Dependency, Phase, PhaseManager, Shell, Stage, StageMain}
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}

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
        Dependency[circt.stage.phases.Checks],
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

  /** Return a CHIRRTL circuit for a Chisel module
    *
    * @param gen a call-by-name Chisel module
    */
  def convert(gen: => RawModule): firrtl.ir.Circuit = {
    chisel3.stage.ChiselStage.convert(gen)
  }

  /** A phase shared by all the CIRCT backends */
  private def phase = new PhaseManager(
    Seq(
      Dependency[chisel3.stage.phases.Checks],
      Dependency[chisel3.aop.injecting.InjectingPhase],
      Dependency[chisel3.stage.phases.Elaborate],
      Dependency[chisel3.stage.phases.Convert],
      Dependency[firrtl.stage.phases.AddImplicitOutputFile],
      Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
      Dependency[circt.stage.phases.Checks],
      Dependency[circt.stage.phases.CIRCT]
    )
  )

  /** Compile a Chisel circuit to FIRRTL dialect */
  def emitFIRRTLDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.FIRRTL)
      )
    )
    .collectFirst {
      case EmittedMLIR(_, a, _) => a
    }
    .get

  /** Compile a Chisel circuit to HWS dialect */
  def emitHWDialect(gen: => RawModule): String = phase
    .transform(
      Seq(
        ChiselGeneratorAnnotation(() => gen),
        CIRCTTargetAnnotation(CIRCTTarget.HW)
      )
    )
    .collectFirst {
      case EmittedMLIR(_, a, _) => a
    }
    .get

  /** Compile a Chisel circuit to SystemVerilog
    *
    * @param gen         a call-by-name Chisel module
    * @param args        additional command line arguments to pass to Chisel
    * @param firtoolOpts additional [[circt.stage.FirtoolOption]] to pass to firtool
    * @return a string containing the Verilog output
    */
  def emitSystemVerilog(
    gen:         => RawModule,
    args:        Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  ): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog)
    ) ++ (new circt.stage.ChiselStage).shell.parse(args) ++ firtoolOpts.map(FirtoolOption(_))
    phase
      .transform(annos)
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }
      .get
      .value
  }

  /** Compile a Chisel circuit to SystemVerilog with file output
    *
    * @param gen         a call-by-name Chisel module
    * @param args        additional command line arguments to pass to Chisel
    * @param firtoolOpts additional command line options to pass to firtool
    * @return a string containing the Verilog output
    */
  def emitSystemVerilogFile(
    gen:         => RawModule,
    args:        Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  ) = {
    val chiselArgs = Array("--target", "systemverilog") ++ args
    (new circt.stage.ChiselStage).execute(
      chiselArgs,
      Seq(ChiselGeneratorAnnotation(() => gen)) ++ firtoolOpts.map(FirtoolOption(_))
    )
  }

  /** Return a Chisel circuit for a Chisel module
    *
    * @param gen a call-by-name Chisel module
    */
  def elaborate(
    gen: => RawModule
  ): chisel3.internal.firrtl.Circuit = {
    chisel3.stage.ChiselStage.elaborate(gen)
  }
}

/** Command line entry point to [[ChiselStage]] */
object ChiselMain extends StageMain(new ChiselStage)
