// SPDX-License-Identifier: Apache-2.0

package chisel3.stage

import chisel3.internal.firrtl.ir.Circuit
import chisel3.internal.{AutoTestIncluder, WarningFilter}
import chisel3.layer.Layer
import chisel3.ElaboratedCircuit
import java.io.File
import scala.annotation.nowarn

class ChiselOptions private[stage] (
  val printFullStackTrace: Boolean = false,
  val throwOnFirstError:   Boolean = false,
  val outputFile:          Option[String] = None,
  val sourceRoots:         Vector[File] = Vector.empty,
  val warningFilters:      Vector[WarningFilter] = Vector.empty,
  val useLegacyWidth:      Boolean = false,
  val layerMap:            Map[Layer, Layer] = Map.empty,
  val includeUtilMetadata: Boolean = false,
  val useSRAMBlackbox:     Boolean = false,
  val elaboratedCircuit:   Option[ElaboratedCircuit] = None,
  val autoTestIncluder:    AutoTestIncluder = AutoTestIncluder.none,
  val suppressSourceInfo:  Boolean = false
) {

  /** The old auto-test includer name. */
  @deprecated("Use autoTestIncluder", "Chisel 7.16.0")
  @nowarn("cat=deprecation")
  def inlineTestIncluder: chisel3.internal.InlineTestIncluder = autoTestIncluder

  private[stage] def copy(
    printFullStackTrace: Boolean = printFullStackTrace,
    throwOnFirstError:   Boolean = throwOnFirstError,
    outputFile:          Option[String] = outputFile,
    sourceRoots:         Vector[File] = sourceRoots,
    warningFilters:      Vector[WarningFilter] = warningFilters,
    useLegacyWidth:      Boolean = useLegacyWidth,
    layerMap:            Map[Layer, Layer] = layerMap,
    includeUtilMetadata: Boolean = includeUtilMetadata,
    useSRAMBlackbox:     Boolean = useSRAMBlackbox,
    elaboratedCircuit:   Option[ElaboratedCircuit] = elaboratedCircuit,
    autoTestIncluder:    AutoTestIncluder = autoTestIncluder,
    suppressSourceInfo:  Boolean = suppressSourceInfo
  ): ChiselOptions = {

    new ChiselOptions(
      printFullStackTrace = printFullStackTrace,
      throwOnFirstError = throwOnFirstError,
      outputFile = outputFile,
      sourceRoots = sourceRoots,
      warningFilters = warningFilters,
      useLegacyWidth = useLegacyWidth,
      layerMap = layerMap,
      includeUtilMetadata = includeUtilMetadata,
      useSRAMBlackbox = useSRAMBlackbox,
      elaboratedCircuit = elaboratedCircuit,
      autoTestIncluder = autoTestIncluder,
      suppressSourceInfo = suppressSourceInfo
    )

  }

}
