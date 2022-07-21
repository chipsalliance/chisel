// SPDX-License-Identifier: Apache-2.0

package circt.stage

import java.io.File

/** Options associated with CIRCT
  *
  * @param inputFile the name of an input FIRRTL IR file
  * @param outputFile the name of the file where the result will be written
  * @param preserveAggregate causes CIRCT to not lower aggregate FIRRTL IR types
  * @param target the specific IR or language target that CIRCT should compile to
  */
class CIRCTOptions private[stage] (
  val inputFile:         Option[File] = None,
  val outputFile:        Option[File] = None,
  val preserveAggregate: Option[PreserveAggregate.Type] = None,
  val target:            Option[CIRCTTarget.Type] = None,
  val handover:          Option[CIRCTHandover.Type] = None,
  val firtoolOptions:    Seq[String] = Seq.empty) {

  private[stage] def copy(
    inputFile:         Option[File] = inputFile,
    outputFile:        Option[File] = outputFile,
    preserveAggregate: Option[PreserveAggregate.Type] = preserveAggregate,
    target:            Option[CIRCTTarget.Type] = target,
    handover:          Option[CIRCTHandover.Type] = handover,
    firtoolOptions:    Seq[String] = firtoolOptions
  ): CIRCTOptions = new CIRCTOptions(inputFile, outputFile, preserveAggregate, target, handover, firtoolOptions)

}
