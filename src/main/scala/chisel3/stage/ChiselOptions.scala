// See LICENSE for license details.

package chisel3.stage

import chisel3.internal.firrtl.Circuit

class ChiselOptions private[stage] (
  val runFirrtlCompiler:   Boolean         = true,
  val printFullStackTrace: Boolean         = false,
  val outputFile:          Option[String]  = None,
  val chiselCircuit:       Option[Circuit] = None) {

  private[stage] def copy(
    runFirrtlCompiler:   Boolean         = runFirrtlCompiler,
    printFullStackTrace: Boolean         = printFullStackTrace,
    outputFile:          Option[String]  = outputFile,
    chiselCircuit:       Option[Circuit] = chiselCircuit ): ChiselOptions = {

    new ChiselOptions(
      runFirrtlCompiler   = runFirrtlCompiler,
      printFullStackTrace = printFullStackTrace,
      outputFile          = outputFile,
      chiselCircuit       = chiselCircuit )

  }

}
