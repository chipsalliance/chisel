// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.WarningFilter
import chisel3.layer.Layer
import chisel3.ElaboratedCircuit
import java.io.File

class ChiselOptions private[stage] (
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val outputFile:          Option[String] = None,
  _chiselCircuit:          Option[Circuit] = None,
  val sourceRoots:         Vector[File] = Vector.empty,
  val warningFilters:      Vector[WarningFilter] = Vector.empty,
  val useLegacyWidth:      Boolean = false,
  val layerMap:            Map[Layer, Layer] = Map.empty,
  val includeUtilMetadata: Boolean = false,
  val useSRAMBlackbox:     Boolean = false,
  val elaboratedCircuit:   Option[ElaboratedCircuit] = None
) {

  @deprecated("Use elaboratedCircuit instead", "Chisel 6.7.0")
  def chiselCircuit: Option[Circuit] = _chiselCircuit

  private[stage] def copy(
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    outputFile:          Option[String] = outputFile,
    chiselCircuit:       Option[Circuit] = _chiselCircuit,
    sourceRoots:         Vector[File] = sourceRoots,
    warningFilters:      Vector[WarningFilter] = warningFilters,
    useLegacyWidth:      Boolean = useLegacyWidth,
    layerMap:            Map[Layer, Layer] = layerMap,
    includeUtilMetadata: Boolean = includeUtilMetadata,
    useSRAMBlackbox:     Boolean = useSRAMBlackbox,
    elaboratedCircuit:   Option[ElaboratedCircuit] = elaboratedCircuit
  ): ChiselOptions = {

    new ChiselOptions(
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      outputFile = outputFile,
      _chiselCircuit = _chiselCircuit,
      sourceRoots = sourceRoots,
      warningFilters = warningFilters,
      useLegacyWidth = useLegacyWidth,
      layerMap = layerMap,
      includeUtilMetadata = includeUtilMetadata,
      useSRAMBlackbox = useSRAMBlackbox,
      elaboratedCircuit = elaboratedCircuit
    )

  }

}
