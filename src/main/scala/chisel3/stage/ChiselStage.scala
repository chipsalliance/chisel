// See LICENSE for license details.

package chisel3.stage

import firrtl.{
  ir => fir,
  AnnotationSeq,
  EmittedFirrtlCircuitAnnotation,
  EmittedVerilogCircuitAnnotation,
  HighFirrtlEmitter,
  VerilogEmitter,
  SystemVerilogEmitter
}
import firrtl.options.{Dependency, Phase, PhaseManager, Shell, Stage, StageError, StageMain}
import firrtl.options.phases.DeletedWrapper
import firrtl.stage.{FirrtlCircuitAnnotation, FirrtlCli, RunFirrtlTransformAnnotation}
import firrtl.options.Viewer.view

import chisel3.{ChiselException, RawModule}
import chisel3.internal.{firrtl => cir, ErrorLog}
import chisel3.stage.CircuitSerializationAnnotation.FirrtlFileFormat

import java.io.{StringWriter, PrintWriter}

class ChiselStage extends Stage {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  val shell: Shell = new Shell("chisel") with ChiselCli with FirrtlCli

  val targets: Seq[PhaseManager.PhaseDependency] = ChiselPhase.targets

  final lazy val phaseManager = {
    val _targets = targets
    new ChiselPhase {
      override val targets = _targets
    }
  }

  def run(annotations: AnnotationSeq): AnnotationSeq = try {
    phaseManager.transform(annotations)
  } catch {
    case ce: ChiselException =>
      val stackTrace = if (!view[ChiselOptions](annotations).printFullStackTrace) {
        ce.chiselStackTrace
      } else {
        val sw = new StringWriter
        ce.printStackTrace(new PrintWriter(sw))
        sw.toString
      }
      Predef
        .augmentString(stackTrace)
        .lines
        .foreach(line => println(s"${ErrorLog.errTag} $line"))
      throw new StageError(cause=ce)
  }

  /** Convert a Chisel module to a CHIRRTL string
    * @param gen a call-by-name Chisel module
    * @param args additional command line arguments to pass to Chisel
    * param annotations additional annotations to pass to Chisel
    * @return a string containing the Verilog output
    */
  final def emitChirrtl(
    gen: => RawModule,
    args: Array[String] = Array.empty,
    annotations: AnnotationSeq = Seq.empty): String = {

    val annos = execute(Array("--no-run-firrtl") ++ args, ChiselGeneratorAnnotation(() => gen) +: annotations)

    annos
      .collectFirst {
        case a: ChiselCircuitAnnotation => CircuitSerializationAnnotation(a.circuit, "", FirrtlFileFormat).getBytes
      }
      .get
      .map(_.toChar)
      .mkString

  }

  /** Convert a Chisel module to a FIRRTL string
    * @param gen a call-by-name Chisel module
    * @param args additional command line arguments to pass to Chisel
    * param annotations additional annotations to pass to Chisel
    * @return a string containing the FIRRTL output
    */
  final def emitFirrtl(
    gen: => RawModule,
    args: Array[String] = Array.empty,
    annotations: AnnotationSeq = Seq.empty): String = {

    execute(Array("-X", "high") ++ args, ChiselGeneratorAnnotation(() => gen) +: annotations)
      .collectFirst {
        case EmittedFirrtlCircuitAnnotation(a) => a
      }
      .get
      .value

  }

  /** Convert a Chisel module to Verilog
    * @param gen a call-by-name Chisel module
    * @param args additional command line arguments to pass to Chisel
    * param annotations additional annotations to pass to Chisel
    * @return a string containing the Verilog output
    */
  final def emitVerilog(
    gen: => RawModule,
    args: Array[String] = Array.empty,
    annotations: AnnotationSeq = Seq.empty): String = {

    execute(Array("-X", "verilog") ++ args, ChiselGeneratorAnnotation(() => gen) +: annotations)
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }
      .get
      .value
  }

  /** Convert a Chisel module to SystemVerilog
    * @param gen a call-by-name Chisel module
    * @param args additional command line arguments to pass to Chisel
    * param annotations additional annotations to pass to Chisel
    * @return a string containing the SystemVerilog output
    */
  final def emitSystemVerilog(
                         gen: => RawModule,
                         args: Array[String] = Array.empty,
                         annotations: AnnotationSeq = Seq.empty): String = {

    execute(Array("-X", "sverilog") ++ args, ChiselGeneratorAnnotation(() => gen) +: annotations)
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }
      .get
      .value
  }
}

object ChiselMain extends StageMain(new ChiselStage)

/** Helper methods for working with [[ChiselStage]] */
object ChiselStage {

  /** Return a Chisel circuit for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def elaborate(gen: => RawModule): cir.Circuit = {
    val phase = new ChiselPhase {
      override val targets = Seq( Dependency[chisel3.stage.phases.Checks],
                                  Dependency[chisel3.stage.phases.Elaborate] )
    }

    phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen), NoRunFirrtlCompilerAnnotation))
      .collectFirst {
        case ChiselCircuitAnnotation(a) => a
      }
      .get
  }

  /** Return a CHIRRTL circuit for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def convert(gen: => RawModule): fir.Circuit = {
    val phase = new ChiselPhase {
      override val targets = Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.Convert] )
    }

    phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen)))
      .collectFirst {
        case FirrtlCircuitAnnotation(a) => a
      }
      .get
  }

  /** Return a CHIRRTL string for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def emitChirrtl(gen: => RawModule): String = convert(gen).serialize

  /** Return a FIRRTL string for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def emitFirrtl(gen: => RawModule): String = {
    val phase = new PhaseManager(
      Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[firrtl.stage.phases.Compiler] )
    )

    phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen), RunFirrtlTransformAnnotation(new HighFirrtlEmitter)))
      .collectFirst {
        case EmittedFirrtlCircuitAnnotation(a) => a
      }.get
      .value

  }

  /** Return a Verilog string for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def emitVerilog(gen: => RawModule): String = {
    val phase = new PhaseManager(
      Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[firrtl.stage.phases.Compiler] )
    )

    phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen), RunFirrtlTransformAnnotation(new VerilogEmitter)))
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }.get
      .value
  }

  /** Return a SystemVerilog string for a Chisel module
    * @param gen a call-by-name Chisel module
    */
  def emitSystemVerilog(gen: => RawModule): String = {
    val phase = new PhaseManager(
      Seq(
        Dependency[chisel3.stage.phases.Checks],
        Dependency[chisel3.stage.phases.Elaborate],
        Dependency[chisel3.stage.phases.AddImplicitOutputFile],
        Dependency[chisel3.stage.phases.AddImplicitOutputAnnotationFile],
        Dependency[chisel3.stage.phases.MaybeAspectPhase],
        Dependency[chisel3.stage.phases.Convert],
        Dependency[firrtl.stage.phases.Compiler] )
    )

    phase
      .transform(Seq(ChiselGeneratorAnnotation(() => gen), RunFirrtlTransformAnnotation(new SystemVerilogEmitter)))
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }.get
      .value
  }

}
