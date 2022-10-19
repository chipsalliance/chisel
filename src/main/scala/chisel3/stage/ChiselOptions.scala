// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.Circuit

class ChiselOptions private[stage] (
  val runFirrtlCompiler:      Boolean = true,
  val printFullStackTrace:    Boolean = false,
  val omitSourceLocatorPaths: Boolean = false,
  val throwOnFirstError:      Boolean = false,
  val warningsAsErrors:       Boolean = false,
  val outputFile:             Option[String] = None,
  val chiselCircuit:          Option[Circuit] = None) {

  private[stage] def copy(
    runFirrtlCompiler:      Boolean = runFirrtlCompiler,
    printFullStackTrace:    Boolean = printFullStackTrace,
    omitSourceLocatorPaths: Boolean = omitSourceLocatorPaths,
    throwOnFirstError:      Boolean = throwOnFirstError,
    warningsAsErrors:       Boolean = warningsAsErrors,
    outputFile:             Option[String] = outputFile,
    chiselCircuit:          Option[Circuit] = chiselCircuit
  ): ChiselOptions = {

    new ChiselOptions(
      runFirrtlCompiler = runFirrtlCompiler,
      printFullStackTrace = printFullStackTrace,
      omitSourceLocatorPaths = omitSourceLocatorPaths,
      throwOnFirstError = throwOnFirstError,
      warningsAsErrors = warningsAsErrors,
      outputFile = outputFile,
      chiselCircuit = chiselCircuit
    )

  }

}
