package chisel3

import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{CIRCTTarget, CIRCTTargetAnnotation, ChiselStage}
import firrtl.{AnnotationSeq, EmittedVerilogCircuitAnnotation}
import firrtl.options.{Dependency, PhaseManager}

object getVerilogString {

  final def phase = new PhaseManager(
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

  /**
    * Returns a string containing the Verilog for the module specified by
    * the target.
    *
    * @param gen the module to be converted to Verilog
    * @return a string containing the Verilog for the module specified by
    *         the target
    */
  def apply(gen: => RawModule): String = ChiselStage.emitSystemVerilog(gen)

  /**
    * Returns a string containing the Verilog for the module specified by
    * the target accepting arguments and annotations
    *
    * @param gen the module to be converted to Verilog
    * @param args arguments to be passed to the compiler
    * @param annotations annotations to be passed to the compiler
    * @return a string containing the Verilog for the module specified by
    *         the target
    */
  def apply(gen: => RawModule, args: Array[String] = Array.empty, annotations: AnnotationSeq = Seq.empty): String = {
    val annos = Seq(
      ChiselGeneratorAnnotation(() => gen),
      CIRCTTargetAnnotation(CIRCTTarget.SystemVerilog)
    ) ++ (new circt.stage.ChiselStage).shell.parse(args) ++ annotations
    phase
      .transform(annos)
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }
      .get
      .value
  }
}

object emitVerilog {
  def apply(gen: => RawModule, args: Array[String] = Array.empty, annotations: AnnotationSeq = Seq.empty): String = {
    (new ChiselStage)
      .execute(
        Array("--target", "systemverilog") ++ args,
        ChiselGeneratorAnnotation(() => gen) +: annotations
      )
      .collectFirst {
        case EmittedVerilogCircuitAnnotation(a) => a
      }
      .get
      .value
  }
}
