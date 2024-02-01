// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

@deprecated("There no CIRCTPassManager anymore, use circtpanamaconverter directly", "Chisel 6.0")
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
