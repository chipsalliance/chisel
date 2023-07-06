// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import chisel3.experimental.{SourceInfo, SourceLineNoCol, UnlocatableSourceInfo}

///////////////////////////////////////////////////
// Never remove IDs and only ever add to the end //
///////////////////////////////////////////////////

// TODO should deprecations be included here?
private[chisel3] object WarningID extends Enumeration {
  type WarningID = Value

  val NoID = Value(0) // Reserved
  val UnsafeUIntCastToEnum = Value(1)
  val DynamicBitSelectTooWide = Value(2)
  val DynamicBitSelectTooNarrow = Value(3)
  val DynamicIndexTooWide = Value(4)
  val DynamicIndexTooNarrow = Value(5)
  val ExtractFromVecSizeZero = Value(6)
}
import WarningID.WarningID

// Argument order differs from apply below to avoid type signature collision with apply method below
private[chisel3] case class Warning(info: SourceInfo, id: WarningID, msg: String)
private[chisel3] object Warning {
  def apply(id: WarningID, msg: String)(implicit info: SourceInfo): Warning = {
    val num = f"[W${id.id}%03d] "
    new Warning(info, id, num + msg)
  }
  def noInfo(id: WarningID, msg: String): Warning = {
    implicit val info = SourceLineNoCol.materialize.getOrElse(UnlocatableSourceInfo)
    Warning(id, msg)
  }
}
