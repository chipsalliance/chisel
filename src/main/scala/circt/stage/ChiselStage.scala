// SPDX-License-Identifier: Apache-2.0

package circt.stage

import chisel3.RawModule
import chisel3.stage.{ChiselCircuitAnnotation, ChiselGeneratorAnnotation, CircuitSerializationAnnotation}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import firrtl.options.{CustomFileEmission, Dependency, Phase, PhaseManager, Stage, StageMain, Unserializable}
import firrtl.stage.FirrtlCircuitAnnotation
import logger.LogLevelAnnotation
import firrtl.EmittedBtor2CircuitAnnotation

/** Entry point for running Chisel with the CIRCT compiler.
  *
  * @note The companion object, [[ChiselStage$]], has a cleaner API for compiling and returning a string.
  */
class ChiselStage extends Stage {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override val shell = new firrtl.options.Shell("circt") with CLI {
    // These are added by firrtl.options.Shell (which we must extend because we are a Stage)
    override protected def includeLoggerOptions = false
  }

  override def run(annotations: AnnotationSeq): AnnotationSeq = {

    val pm = new PhaseManager(
      targets = Seq(
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.AddSerializationAnnotations],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[chisel3.stage.phases.AddDedupGroupAnnotations],
        Dependency[circt.stage.phases.AddImplicitOutputFile],
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
    ) ++ (new Shell("circt")).parse(args)

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

  /** Elaborates a Chisel circuit and emits it to a file
    *
    * @param gen  a call-by-name Chisel module
    * @param args additional command line arguments to pass to Chisel
    */
  def emitCHIRRTLFile(
    gen:  => RawModule,
    args: Array[String] = Array.empty
  ): AnnotationSeq = {
    (new circt.stage.ChiselStage).execute(
      Array("--target", "chirrtl") ++ args,
      Seq(ChiselGeneratorAnnotation(() => gen))
    )
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
    ) ++ (new Shell("circt")).parse(args)

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
    ) ++ (new Shell("circt")).parse(args) ++ firtoolOpts.map(FirtoolOption(_))

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
    ) ++ (new Shell("circt")).parse(args) ++ firtoolOpts.map(FirtoolOption(_))

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
    ) ++ (new Shell("circt")).parse(args) ++ firtoolOpts.map(FirtoolOption(_))
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
  ): AnnotationSeq =
    (new circt.stage.ChiselStage).execute(
      Array("--target", "systemverilog") ++ args,
      Seq(ChiselGeneratorAnnotation(() => gen)) ++ firtoolOpts.map(FirtoolOption(_))
    )

  /** Compile a Chisel circuit to btor2
    *
    * @param gen         a call-by-name Chisel module
    * @param args        additional command line arguments to pass to Chisel
    * @param firtoolOpts additional command line options to pass to firtool
    * @return a string containing the btor2 output
    */
  def emitBtor2(
    gen:         => RawModule,
    args:        Array[String] = Array.empty,
    firtoolOpts: Array[String] = Array.empty
  ): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.Btor2)
    ) ++ (new Shell("circt")).parse(args) ++ firtoolOpts.map(FirtoolOption(_))
    phase
      .transform(annos)
      .collectFirst {
        case EmittedBtor2CircuitAnnotation(a) => a
      }
      .get
      .value
  }
}

/** Command line entry point to [[ChiselStage]] */
object ChiselMain extends StageMain(new ChiselStage)
