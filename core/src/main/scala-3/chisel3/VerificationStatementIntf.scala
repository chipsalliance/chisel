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
    using sourceInfo: SourceInfo
  ): Assert = VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(
    cond,
    emptySourceLine,
    Some(Printable.pack(message, data: _*))
  )

  def apply(
    cond:    Bool,
    message: Printable
  )(
    using sourceInfo: SourceInfo
  ): Assert = VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(cond, emptySourceLine, Some(message))

  def apply(cond: Bool)(using sourceInfo: SourceInfo): Assert =
    VerifStmtMacrosCompat.assert._applyWithSourceLinePrintable(cond, emptySourceLine, None)
}

private[chisel3] trait Assume$Intf extends VerifPrintMacrosDoc { self: assume.type =>

  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    using sourceInfo: SourceInfo
  ): Assume = VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(
    cond,
    emptySourceLine,
    Some(Printable.pack(message, data: _*))
  )

  def apply(
    cond:    Bool,
    message: Printable
  )(
    using sourceInfo: SourceInfo
  ): Assume = VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(cond, emptySourceLine, Some(message))

  def apply(cond: Bool)(using sourceInfo: SourceInfo): Assume =
    VerifStmtMacrosCompat.assume._applyWithSourceLinePrintable(cond, emptySourceLine, None)
}

private[chisel3] trait Cover$Impl extends VerifPrintMacrosDoc { self: cover.type =>
  def apply(cond: Bool, message: String)(using sourceInfo: SourceInfo): Cover =
    VerifStmtMacrosCompat.cover._applyWithSourceLine(cond, emptySourceLine, Some(message))
  def apply(cond: Bool)(using sourceInfo: SourceInfo): Cover =
    VerifStmtMacrosCompat.cover._applyWithSourceLine(cond, emptySourceLine, None)
}

private[chisel3] trait Stop$Intf { self: stop.type =>

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a message describing why simulation was stopped
    */
  def apply(message: String)(using sourceInfo: SourceInfo): Stop = _applyImpl(message)

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a printable describing why simulation was stopped
    */
  def apply(message: Printable)(using sourceInfo: SourceInfo): Stop = _applyImpl(message)

  /** Terminate execution, indicating success.
    */
  def apply()(using sourceInfo: SourceInfo): Stop = _applyImpl()
}
