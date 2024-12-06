// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.WarningFilter
import chisel3.layer.Layer
import java.io.File

class ChiselOptions private[stage] (
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val outputFile:          Option[String] = None,
  val chiselCircuit:       Option[Circuit] = None,
  val sourceRoots:         Vector[File] = Vector.empty,
  val warningFilters:      Vector[WarningFilter] = Vector.empty,
  val useLegacyWidth:      Boolean = false,
  val layerMap:            Map[Layer, Layer] = Map.empty,
  val includeUtilMetadata: Boolean = false,
  val useSRAMBlackbox:     Boolean = false) {

  private[stage] def copy(
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    outputFile:          Option[String] = outputFile,
    chiselCircuit:       Option[Circuit] = chiselCircuit,
    sourceRoots:         Vector[File] = sourceRoots,
    warningFilters:      Vector[WarningFilter] = warningFilters,
    useLegacyWidth:      Boolean = useLegacyWidth,
    layerMap:            Map[Layer, Layer] = layerMap,
    includeUtilMetadata: Boolean = includeUtilMetadata,
    useSRAMBlackbox:     Boolean = useSRAMBlackbox
  ): ChiselOptions = {

    new ChiselOptions(
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      outputFile = outputFile,
      chiselCircuit = chiselCircuit,
      sourceRoots = sourceRoots,
      warningFilters = warningFilters,
      useLegacyWidth = useLegacyWidth,
      layerMap = layerMap,
      includeUtilMetadata = includeUtilMetadata,
      useSRAMBlackbox = useSRAMBlackbox
    )

  }

}
