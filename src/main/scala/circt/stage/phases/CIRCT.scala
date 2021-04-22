// SPDX-License-Identifier: Apache-2.0

package circt.stage.phases

import circt.Implicits.BooleanImplicits
import circt.stage.{
  CIRCTOptions,
  CIRCTTarget,
  EmittedMLIR
}

import firrtl.{
  AnnotationSeq,
  EmittedVerilogCircuit,
  EmittedVerilogCircuitAnnotation
}
import firrtl.options.{
  Dependency,
  OptionsException,
  Phase,
  StageError,
  StageOptions,
  StageUtils
}
import firrtl.options.Viewer.view
import firrtl.stage.FirrtlOptions

import scala.sys.process._

/** A phase that calls and runs CIRCT, specifically `firtool`, while preserving an [[AnnotationSeq]] API.
  *
  * This is analogous to [[firrtl.stage.phases.Compiler]].
  */
class CIRCT extends Phase {

  override def prerequisites = Seq(
    Dependency[firrtl.stage.phases.AddDefaults],
    Dependency[firrtl.stage.phases.AddImplicitEmitter],
    Dependency[firrtl.stage.phases.AddImplicitOutputFile]
  )
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf = Seq.empty
  override def invalidates(a: Phase) = false

  override def transform(annotations: AnnotationSeq): AnnotationSeq = {
    val circtOptions = view[CIRCTOptions](annotations)
    val firrtlOptions = view[FirrtlOptions](annotations)
    val stageOptions = view[StageOptions](annotations)

    val input: String = firrtlOptions.firrtlCircuit match {
      case None => throw new OptionsException("No input file specified!")
      case Some(circuit) => circuit.serialize
    }

    val outputFileName: String = stageOptions.getBuildFileName(firrtlOptions.outputFileName.get)

    val binary = "firtool"

    val cmd =
      Seq(binary, "-format=fir") ++
        (!circtOptions.disableLowerTypes).option("-lower-types") ++
        (circtOptions.target match {
           case Some(CIRCTTarget.FIRRTL) => None
           case Some(CIRCTTarget.RTL) => Seq("-lower-to-rtl")
           case Some(CIRCTTarget.Verilog) => Seq("-lower-to-rtl", "-verilog")
           case Some(CIRCTTarget.SystemVerilog) => Seq("-lower-to-rtl", "-verilog")
           case None => throw new Exception(
             "No 'circtOptions.target' specified. This should be impossible if dependencies are satisfied!"
           )
         })

    try {
      logger.info(s"""Running CIRCT: '${cmd.mkString(" ")} < $$input'""")
      val result = (cmd #< new java.io.ByteArrayInputStream(input.getBytes)).!!

      circtOptions.target match {
        case Some(CIRCTTarget.FIRRTL) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".fir.mlir")))
        case Some(CIRCTTarget.RTL) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".rtl.mlir")))
        case Some(CIRCTTarget.Verilog) =>
          Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(outputFileName, result, ".v")))
        case Some(CIRCTTarget.SystemVerilog) =>
          Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(outputFileName, result, ".sv")))
        case None => throw new Exception(
          "No 'circtOptions.target' specified. This should be impossible if dependencies are satisfied!"
        )
      }
    } catch {
      case a: java.io.IOException =>
        StageUtils.dramaticError(s"Binary '$binary' was not found on the $$PATH. (Do you have CIRCT installed?)")
        throw new StageError(cause = a)
    }

  }

}
