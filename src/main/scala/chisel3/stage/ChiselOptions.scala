// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.Circuit
import chisel3.internal.WarningFilter
import java.io.File

class ChiselOptions private[stage] (
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val outputFile:          Option[String] = None,
  val chiselCircuit:       Option[Circuit] = None,
  val sourceRoots:         Vector[File] = Vector.empty,
  val warningFilters:      Vector[WarningFilter] = Vector.empty) {

  private[stage] def copy(
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    outputFile:          Option[String] = outputFile,
    chiselCircuit:       Option[Circuit] = chiselCircuit,
    sourceRoots:         Vector[File] = sourceRoots,
    warningFilters:      Vector[WarningFilter] = warningFilters
  ): ChiselOptions = {

    new ChiselOptions(
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      outputFile = outputFile,
      chiselCircuit = chiselCircuit,
      sourceRoots = sourceRoots,
      warningFilters = warningFilters
    )

  }

}
