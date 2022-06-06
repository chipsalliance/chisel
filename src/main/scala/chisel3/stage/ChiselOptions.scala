// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.Circuit

class ChiselOptions private[stage] (
  val runFirrtlCompiler:    Boolean = true,
  val printFullStackTrace:  Boolean = false,
  val throwOnFirstError:    Boolean = false,
  val warnReflectiveNaming: Boolean = false,
  val outputFile:           Option[String] = None,
  val chiselCircuit:        Option[Circuit] = None) {

  private[stage] def copy(
    runFirrtlCompiler:    Boolean = runFirrtlCompiler,
    printFullStackTrace:  Boolean = printFullStackTrace,
    throwOnFirstError:    Boolean = throwOnFirstError,
    warnReflectiveNaming: Boolean = warnReflectiveNaming,
    outputFile:           Option[String] = outputFile,
    chiselCircuit:        Option[Circuit] = chiselCircuit
  ): ChiselOptions = {

    new ChiselOptions(
      runFirrtlCompiler = runFirrtlCompiler,
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      warnReflectiveNaming = warnReflectiveNaming,
      outputFile = outputFile,
      chiselCircuit = chiselCircuit
    )

  }

}
