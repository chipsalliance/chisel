// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.WarningFilter
<<<<<<< HEAD
||||||| parent of 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
import chisel3.layer.Layer
=======
import chisel3.layer.Layer
import chisel3.ElaboratedCircuit
>>>>>>> 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
import java.io.File

class ChiselOptions private[stage] (
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val outputFile:          Option[String] = None,
  _chiselCircuit:          Option[Circuit] = None,
  val sourceRoots:         Vector[File] = Vector.empty,
<<<<<<< HEAD
  val warningFilters:      Vector[WarningFilter] = Vector.empty) {
||||||| parent of 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
  val warningFilters:      Vector[WarningFilter] = Vector.empty,
  val useLegacyWidth:      Boolean = false,
  val layerMap:            Map[Layer, Layer] = Map.empty,
  val includeUtilMetadata: Boolean = false,
  val useSRAMBlackbox:     Boolean = false
) {
=======
  val warningFilters:      Vector[WarningFilter] = Vector.empty,
  val useLegacyWidth:      Boolean = false,
  val layerMap:            Map[Layer, Layer] = Map.empty,
  val includeUtilMetadata: Boolean = false,
  val useSRAMBlackbox:     Boolean = false,
  val elaboratedCircuit:   Option[ElaboratedCircuit] = None
) {
>>>>>>> 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))

  @deprecated("Use elaboratedCircuit instead", "Chisel 6.7.0")
  def chiselCircuit: Option[Circuit] = _chiselCircuit

  private[stage] def copy(
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    outputFile:          Option[String] = outputFile,
    chiselCircuit:       Option[Circuit] = _chiselCircuit,
    sourceRoots:         Vector[File] = sourceRoots,
<<<<<<< HEAD
    warningFilters:      Vector[WarningFilter] = warningFilters
||||||| parent of 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
    warningFilters:      Vector[WarningFilter] = warningFilters,
    useLegacyWidth:      Boolean = useLegacyWidth,
    layerMap:            Map[Layer, Layer] = layerMap,
    includeUtilMetadata: Boolean = includeUtilMetadata,
    useSRAMBlackbox:     Boolean = useSRAMBlackbox
=======
    warningFilters:      Vector[WarningFilter] = warningFilters,
    useLegacyWidth:      Boolean = useLegacyWidth,
    layerMap:            Map[Layer, Layer] = layerMap,
    includeUtilMetadata: Boolean = includeUtilMetadata,
    useSRAMBlackbox:     Boolean = useSRAMBlackbox,
    elaboratedCircuit:   Option[ElaboratedCircuit] = elaboratedCircuit
>>>>>>> 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
  ): ChiselOptions = {

    new ChiselOptions(
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      outputFile = outputFile,
      _chiselCircuit = _chiselCircuit,
      sourceRoots = sourceRoots,
<<<<<<< HEAD
      warningFilters = warningFilters
||||||| parent of 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
      warningFilters = warningFilters,
      useLegacyWidth = useLegacyWidth,
      layerMap = layerMap,
      includeUtilMetadata = includeUtilMetadata,
      useSRAMBlackbox = useSRAMBlackbox
=======
      warningFilters = warningFilters,
      useLegacyWidth = useLegacyWidth,
      layerMap = layerMap,
      includeUtilMetadata = includeUtilMetadata,
      useSRAMBlackbox = useSRAMBlackbox,
      elaboratedCircuit = elaboratedCircuit
>>>>>>> 4d755737 (Add ElaboratedCircuit and deprecate use of internal ir Circuit (#4683))
    )

  }

}
