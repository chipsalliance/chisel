// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.panama.circt

import chisel3.internal._

class PanamaCIRCTPassManager private[chisel3] (circt: PanamaCIRCT, mlirModule: MlirModule) extends CIRCTPassManager {
  val pm = circt.mlirPassManagerCreate()
  val options = circt.circtFirtoolOptionsCreateDefault() // TODO: Make it configurable from CIRCTPassManager

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
