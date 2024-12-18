// SPDX-License-Identifier: Apache-2.0

package chisel3.panamaconverter

import chisel3.panamalib._
import chisel3.panamalib.option.FirtoolOptions
import chisel3.panamalib.option.PanamaFirtoolOption.FirtoolOptionsToPanama

private[panamaconverter] class PanamaCIRCTPassManager(
  circt:      PanamaCIRCT,
  mlirModule: MlirModule,
  fos:        Option[FirtoolOptions]
) {
  val pm = circt.mlirPassManagerCreate()
  val options = fos.map(_.toPanama(circt)).getOrElse(circt.circtFirtoolOptionsCreateDefault())

  private def isSuccess(result: MlirLogicalResult): Boolean = circt.mlirLogicalResultIsSuccess(result)

  def populatePreprocessTransforms(): Boolean = isSuccess(circt.circtFirtoolPopulatePreprocessTransforms(pm, options))
  def populateCHIRRTLToLowFIRRTL(): Boolean = isSuccess(
    circt.circtFirtoolPopulateCHIRRTLToLowFIRRTL(pm, options, mlirModule, "-")
  )
  def populateLowFIRRTLToHW(): Boolean = isSuccess(circt.circtFirtoolPopulateLowFIRRTLToHW(pm, options))
  def populateLowHWToSV():     Boolean = isSuccess(circt.circtFirtoolPopulateHWToSV(pm, options))
  def populateExportVerilog(callback: String => Unit): Boolean = isSuccess(
    circt.circtFirtoolPopulateExportVerilog(pm, options, callback)
  )
  def populateExportSplitVerilog(directory: String): Boolean = isSuccess(
    circt.circtFirtoolPopulateExportSplitVerilog(pm, options, directory)
  )
  def populateFinalizeIR(): Boolean = isSuccess(circt.circtFirtoolPopulateFinalizeIR(pm, options))

  def run(): Boolean = isSuccess(circt.mlirPassManagerRunOnOp(pm, circt.mlirModuleGetOperation(mlirModule)))
}
