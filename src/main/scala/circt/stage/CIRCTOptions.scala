// SPDX-License-Identifier: Apache-2.0

package circt.stage

import java.io.File

/** Options associated with CIRCT
  *
  * @param inputFile the name of an input FIRRTL IR file
  * @param outputFile the name of the file where the result will be written
  * @param disableLowerTypes causes CIRCT to not lower aggregate FIRRTL IR types
  * @param target the specific IR or language target that CIRCT should compile to
 */
class CIRCTOptions private[stage](
  val inputFile: Option[File] = None,
  val outputFile: Option[File] = None,
  val disableLowerTypes: Boolean = false,
  val target: Option[CIRCTTarget.Type] = None,
  val handover: Option[CIRCTHandover.Type] = None
) {

  private[stage] def copy(
    inputFile: Option[File] = inputFile,
    outputFile: Option[File] = outputFile,
    disableLowerTypes: Boolean = disableLowerTypes,
    target: Option[CIRCTTarget.Type] = target,
    handover: Option[CIRCTHandover.Type] = handover
  ): CIRCTOptions = new CIRCTOptions(inputFile, outputFile, disableLowerTypes, target, handover)

}
