// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

abstract class CIRCTPassManager {
  def populatePreprocessTransforms(): Boolean
  def populateCHIRRTLToLowFIRRTL():   Boolean
  def populateLowFIRRTLToHW():        Boolean
  def populateLowHWToSV():            Boolean
  def populateExportVerilog(callback:       String => Unit): Boolean
  def populateExportSplitVerilog(directory: String):         Boolean
  def populateFinalizeIR(): Boolean

  def run(): Boolean;
}
