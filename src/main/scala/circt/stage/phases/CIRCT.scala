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

    val cmd = (
      Seq(binary, "-format=fir") ++
        (circtOptions.disableLowerTypes match {
           case true  => None
           case false => Some("-enable-lower-types")
         }) ++
        circtOptions.disableLowerTypes.option("-enable-lower-types") ++
        (circtOptions.target match {
           case Some(CIRCTTarget.FIRRTL) => None
           case Some(CIRCTTarget.RTL) => Seq("-lower-to-rtl")
           case Some(CIRCTTarget.SystemVerilog) => Seq("-lower-to-rtl", "-verilog")
         })
    ) #< new java.io.ByteArrayInputStream(input.getBytes)

    try {
      val result = cmd.!!

      circtOptions.target match {
        case Some(CIRCTTarget.FIRRTL) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".fir.mlir")))
        case Some(CIRCTTarget.RTL) =>
          Seq(EmittedMLIR(outputFileName, result, Some(".rtl.mlir")))
        case Some(CIRCTTarget.SystemVerilog) =>
          Seq(EmittedVerilogCircuitAnnotation(EmittedVerilogCircuit(outputFileName, result, ".sv")))
      }
    } catch {
      case a: java.io.IOException =>
        StageUtils.dramaticError(s"Binary '$binary' was not found on the $$PATH. (Do you have CIRCT installed?)")
        throw new StageError(cause = a)
    }

  }

}
