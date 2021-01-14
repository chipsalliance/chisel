// SPDX-License-Identifier: Apache-2.0

package circt.stage

import java.io.File

class CIRCTOptions private[stage](
  val inputFile: Option[File] = None,
  val outputFile: Option[File] = None,
  val disableLowerTypes: Boolean = false,
  val target: Option[CIRCTTarget.Type] = None
) {

  private[stage] def copy(
    inputFile: Option[File] = inputFile,
    outputFile: Option[File] = outputFile,
    disableLowerTypes: Boolean = disableLowerTypes,
    target: Option[CIRCTTarget.Type] = target
  ): CIRCTOptions = new CIRCTOptions(inputFile, outputFile, disableLowerTypes, target)

}
