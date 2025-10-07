// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}

trait VerifPrintMacrosDoc

private val emptySourceLine: VerifStmtMacrosCompat.SourceLineInfo =
  ("[Source line unavailable: see https://github.com/chipsalliance/chisel/issues/5049]", 0)

private[chisel3] trait Assert$Intf extends VerifPrintMacrosDoc { self: assert.type =>

  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(
    cond,
    emptySourceLine,
    Some(Printable.pack(message, data: _*))
  )

  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(cond, emptySourceLine, Some(message))

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assert =
    VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(cond, emptySourceLine, None)
}

private[chisel3] trait Assume$Intf extends VerifPrintMacrosDoc { self: assume.type =>

  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(
    cond,
    emptySourceLine,
    Some(Printable.pack(message, data: _*))
  )

  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(cond, emptySourceLine, Some(message))

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assume =
    VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(cond, emptySourceLine, None)
}

private[chisel3] trait Cover$Impl extends VerifPrintMacrosDoc { self: cover.type =>
  def apply(cond: Bool, message: String)(implicit sourceInfo: SourceInfo): Cover =
    VerifStmtMacrosCompat.cover._applyWithSourceLine(cond, emptySourceLine, Some(message))
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Cover =
    VerifStmtMacrosCompat.cover._applyWithSourceLine(cond, emptySourceLine, None)
}
