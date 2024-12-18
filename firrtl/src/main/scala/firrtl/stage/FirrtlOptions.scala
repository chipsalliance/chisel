// SPDX-License-Identifier: Apache-2.0

package firrtl.stage

import firrtl.ir.Circuit

/** Internal options used to control the FIRRTL compiler stage.
  * @param outputFileName output file, default: `targetDir/topName.SUFFIX` with `SUFFIX` as determined by the compiler
  * @param infoModeName the policy for generating [[firrtl.ir Info]] when processing FIRRTL (default: "append")
  * @param firrtlCircuit a [[firrtl.ir Circuit]]
  */
class FirrtlOptions private[stage] (
  val outputFileName: Option[String] = None,
  val infoModeName:   String = InfoModeAnnotation().modeName,
  val firrtlCircuit:  Option[Circuit] = None) {

  private[stage] def copy(
    outputFileName: Option[String] = outputFileName,
    infoModeName:   String = infoModeName,
    firrtlCircuit:  Option[Circuit] = firrtlCircuit
  ): FirrtlOptions = {

    new FirrtlOptions(outputFileName = outputFileName, infoModeName = infoModeName, firrtlCircuit = firrtlCircuit)
  }
}
