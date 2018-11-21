// See LICENSE for license details.

package firrtl.stage

import firrtl.{Compiler, Transform}
import firrtl.ir.Circuit

/** Internal options used to control the FIRRTL compiler stage.
  * @param outputFileName output file, default: `targetDir/topName.SUFFIX` with `SUFFIX` as determined by the compiler
  * @param compiler which compiler to use (default: [[VerilogCompiler]])
  * @param infoModeName the policy for generating [[firrtl.ir Info]] when processing FIRRTL (default: "append")
  * @param firrtlCircuit a [[firrtl.ir Circuit]]
  */
class FirrtlOptions private [stage] (
  val outputFileName:       Option[String]  = None,
  val compiler:             Compiler        = CompilerAnnotation().compiler,
  val infoModeName:         String          = InfoModeAnnotation().modeName,
  val firrtlCircuit:        Option[Circuit] = None) {

  private [stage] def copy(
    outputFileName:       Option[String]  = outputFileName,
    compiler:             Compiler        = compiler,
    infoModeName:         String          = infoModeName,
    firrtlCircuit:        Option[Circuit] = firrtlCircuit ): FirrtlOptions = {

    new FirrtlOptions(
      outputFileName       = outputFileName,
      compiler             = compiler,
      infoModeName         = infoModeName,
      firrtlCircuit        = firrtlCircuit )
  }
}
