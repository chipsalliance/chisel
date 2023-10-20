// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.RawModule
import chisel3.stage.{
  ChiselCircuitAnnotation,
  ChiselGeneratorAnnotation,
  CircuitSerializationAnnotation,
  PrintFullStackTraceAnnotation,
  SourceRootAnnotation,
  ThrowOnFirstErrorAnnotation,
  WarningConfigurationAnnotation,
  WarningConfigurationFileAnnotation,
  WarningsAsErrorsAnnotation
}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import firrtl.options.{
  BareShell,
  CustomFileEmission,
  Dependency,
  Phase,
  PhaseManager,
  Shell,
  Stage,
  StageMain,
  Unserializable
}
import firrtl.stage.FirrtlCircuitAnnotation

trait CLI { this: BareShell =>
  parser.note("CIRCT (MLIR FIRRTL Compiler) options")
  Seq(
    CIRCTTargetAnnotation,
    PreserveAggregate,
    ChiselGeneratorAnnotation,
    PrintFullStackTraceAnnotation,
    ThrowOnFirstErrorAnnotation,
    WarningsAsErrorsAnnotation,
    WarningConfigurationAnnotation,
    WarningConfigurationFileAnnotation,
    SourceRootAnnotation,
    SplitVerilog,
    FirtoolBinaryPath,
    DumpFir
  ).foreach(_.addOptions(parser))
}

/** Entry point for running Chisel with the CIRCT compiler.
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
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.AddSerializationAnnotations],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[chisel3.stage.phases.AddDedupGroupAnnotations],
        Dependency[chisel3.stage.phases.MaybeInjectingPhase],
        Dependency[circt.stage.phases.AddImplicitOutputFile],
        Dependency[circt.stage.phases.Checks],
        Dependency[circt.stage.phases.CIRCT]
      ),
      currentState = Seq(
        Dependency[firrtl.stage.phases.AddDefaults],
        Dependency[firrtl.stage.phases.Checks]
      )
    )
    pm.transform(annotations)
  }

}

/** Utilities for compiling Chisel */
object ChiselStage {

  /** A phase shared by all the CIRCT backends */
  private def phase = new PhaseManager(
    Seq(
      Dependency[chisel3.stage.phases.Checks],
      Dependency[chisel3.aop.injecting.InjectingPhase],
      Dependency[chisel3.stage.phases.Elaborate],
      Dependency[chisel3.stage.phases.Convert],
      Dependency[circt.stage.phases.AddImplicitOutputFile],
      Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
      Dependency[circt.stage.phases.Checks],
      Dependency[circt.stage.phases.CIRCT]
    )
  )

  /** Elaborate a Chisel circuit into a CHIRRTL string */
  def emitCHIRRTL(
    gen:  => RawModule,
    args: Array[String] = Array.empty
  ): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL)
    ) ++ (new BareShell("circt") with CLI).parse(args)

    val resultAnnos = phase.transform(annos)

    var circuitAnno: Option[CircuitSerializationAnnotation] = None
    val inFileAnnos = resultAnnos.flatMap {
      case a: ChiselCircuitAnnotation =>
        circuitAnno = Some(CircuitSerializationAnnotation(a.circuit, "", FirrtlFileFormat))
        None
      case _: Unserializable     => None
      case _: CustomFileEmission => None
      case a => Some(a)
    }
    circuitAnno.get.emitLazily(inFileAnnos).mkString
  }

  /** Return a CHIRRTL circuit for a Chisel module
    *
    * @param gen a call-by-name Chisel module
    */
  def convert(
    gen:  => RawModule,
    args: Array[String] = Array.empty
  ): firrtl.ir.Circuit = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL)
    ) ++ (new BareShell("circt") with CLI).parse(args)

    phase
      .transform(annos)
      .collectFirst {
        case FirrtlCircuitAnnotation(a) => a
      }
      .get
  }

  /** Compile a Chisel circuit to FIRRTL dialect */
  def emitFIRRTLDialect(
    gen:         => RawModule,
    args:        Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  ): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.FIRRTL)
    ) ++ (new BareShell("circt") with CLI).parse(args) ++ firtoolOpts.map(FirtoolOption(_))

    phase
      .transform(annos)
      .collectFirst {
        case EmittedMLIR(_, a, _) => a
      }
      .get
  }

  /** Compile a Chisel circuit to HWS dialect */
  def emitHWDialect(
    gen:         => RawModule,
    args:        Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  ): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.HW)
    ) ++ (new BareShell("circt") with CLI).parse(args) ++ firtoolOpts.map(FirtoolOption(_))

    phase
      .transform(annos)
      .collectFirst {
        case EmittedMLIR(_, a, _) => a
      }
      .get
  }

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
    ) ++ (new BareShell("circt") with CLI).parse(args) ++ firtoolOpts.map(FirtoolOption(_))
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
  ) =
    (new circt.stage.ChiselStage).execute(
      Array("--target", "systemverilog") ++ args,
      Seq(ChiselGeneratorAnnotation(() => gen)) ++ firtoolOpts.map(FirtoolOption(_))
    )

  /** Return a Chisel circuit for a Chisel module
    *
    * @param gen a call-by-name Chisel module
    */
  @deprecated(
    "this exposes the internal Chisel circuit which was not supposed to be public---use either ChiselStage.convert or ChiselStage.emitCHIRRTL instead",
    "Chisel 5.0"
  )
  def elaborate(
    gen:  => RawModule,
    args: Array[String] = Array.empty
  ): chisel3.internal.firrtl.Circuit = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL)
    ) ++ (new BareShell("circt") with CLI).parse(args)

    phase
      .transform(annos)
      .collectFirst {
        case ChiselCircuitAnnotation(a) => a
      }
      .get
  }

}

/** Command line entry point to [[ChiselStage]] */
object ChiselMain extends StageMain(new ChiselStage)
