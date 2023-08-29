// SPDX-License-Identifier: Apache-2.0

package circt.stage

import java.io.File

/** Options associated with CIRCT
  *
  * @param inputFile the name of an input FIRRTL IR file
  * @param outputFile the name of the file where the result will be written, if not split
  * @param preserveAggregate causes CIRCT to not lower aggregate FIRRTL IR types
  * @param target the specific IR or language target that CIRCT should compile to
  * @param dumpFir dump the intermediate .fir artifact
  */
class CIRCTOptions private[stage] (
  val outputFile:        Option[File] = None,
  val preserveAggregate: Option[PreserveAggregate.Type] = None,
  val target:            Option[CIRCTTarget.Type] = None,
  val firtoolOptions:    Seq[String] = Seq.empty,
  val splitVerilog:      Boolean = false,
  val firtoolBinaryPath: Option[String] = None,
  val dumpFir:           Boolean = false) {

  private[stage] def copy(
    outputFile:        Option[File] = outputFile,
    preserveAggregate: Option[PreserveAggregate.Type] = preserveAggregate,
    target:            Option[CIRCTTarget.Type] = target,
    firtoolOptions:    Seq[String] = firtoolOptions,
    splitVerilog:      Boolean = splitVerilog,
    firtoolBinaryPath: Option[String] = firtoolBinaryPath,
    dumpFir:           Boolean = dumpFir
  ): CIRCTOptions =
    new CIRCTOptions(outputFile, preserveAggregate, target, firtoolOptions, splitVerilog, firtoolBinaryPath, dumpFir)

}
