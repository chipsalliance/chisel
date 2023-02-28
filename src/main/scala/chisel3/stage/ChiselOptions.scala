// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.Circuit
import java.io.File

class ChiselOptions private[stage] (
  val runFirrtlCompiler:   Boolean = true,
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val warningsAsErrors:    Boolean = false,
  val outputFile:          Option[String] = None,
  val chiselCircuit:       Option[Circuit] = None,
  val sourceRoots:         Vector[File] = Vector.empty) {

  private[stage] def copy(
    runFirrtlCompiler:   Boolean = runFirrtlCompiler,
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    warningsAsErrors:    Boolean = warningsAsErrors,
    outputFile:          Option[String] = outputFile,
    chiselCircuit:       Option[Circuit] = chiselCircuit,
    sourceRoots:         Vector[File] = sourceRoots
  ): ChiselOptions = {

    new ChiselOptions(
      runFirrtlCompiler = runFirrtlCompiler,
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      warningsAsErrors = warningsAsErrors,
      outputFile = outputFile,
      chiselCircuit = chiselCircuit,
      sourceRoots = sourceRoots
    )

  }

}
