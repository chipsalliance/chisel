// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.{BaseModule, SourceInfo}

import scala.language.experimental.macros

/** Scaladoc information for internal verification statement macros
  * that are used in objects assert, assume and cover.
  *
  * @groupdesc VerifPrintMacros
  *
  * <p>
  * '''These internal methods are not part of the public-facing API!'''
  * </p>
  * <br>
  *
  * @groupprio VerifPrintMacros 1001
  */
trait VerifPrintMacrosDoc

private[chisel3] trait Assert$Intf extends VerifPrintMacrosDoc { self: assert.type =>

  /** Checks for a condition to be valid in the circuit at rising clock edge
    * when not in reset. If the condition evaluates to false, the circuit
    * simulation stops with an error.
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional format string to print when the assertion fires
    * @param data optional bits to print in the message formatting
    *
    * @note See [[printf.apply(fmt:String* printf]] for format string documentation
    * @note currently cannot be used in core Chisel / libraries because macro
    * defs need to be compiled first and the SBT project is not set up to do
    * that
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = macro VerifStmtMacrosCompat.assert._applyMacroWithInterpolatorCheck

  /** Checks for a condition to be valid in the circuit at all times. If the
    * condition evaluates to false, the circuit simulation stops with an error.
    *
    * Does not fire when in reset (defined as the current implicit reset, e.g. as set by
    * the enclosing `withReset` or Module.reset.
    *
    * @param cond condition, assertion fires (simulation fails) on a rising clock edge when false and reset is not asserted
    * @param message optional chisel Printable type message
    *
    * @note See [[printf.apply(pable:chisel3\.Printable)*]] for documentation on printf using Printables
    * @note currently cannot be used in core Chisel / libraries because macro
    * defs need to be compiled first and the SBT project is not set up to do
    * that
    */
  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assert = macro VerifStmtMacrosCompat.assert._applyMacroWithPrintableMessage

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assert =
    macro VerifStmtMacrosCompat.assert._applyMacroWithNoMessage
}

private[chisel3] trait Assume$Intf extends VerifPrintMacrosDoc { self: assume.type =>

  /** Assumes a condition to be valid in the circuit at all times.
    * Acts like an assertion in simulation and imposes a declarative
    * assumption on the state explored by formal tools.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using assert make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional format string to print when the assertion fires
    * @param data optional bits to print in the message formatting
    *
    * @note See [[printf.apply(fmt:String* printf]] for format string documentation
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(
    cond:    Bool,
    message: String,
    data:    Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = macro VerifStmtMacrosCompat.assume._applyMacroWithInterpolatorCheck

  /** Assumes a condition to be valid in the circuit at all times.
    * Acts like an assertion in simulation and imposes a declarative
    * assumption on the state explored by formal tools.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * @param cond condition, assertion fires (simulation fails) when false
    * @param message optional Printable type message when the assertion fires
    *
    * @note See [[printf.apply(pable:chisel3\.Printable)*]] for documentation on printf using Printables
    */
  def apply(
    cond:    Bool,
    message: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): Assume = macro VerifStmtMacrosCompat.assume._applyMacroWithPrintableMessage

  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Assume =
    macro VerifStmtMacrosCompat.assume._applyMacroWithNoMessage
}

private[chisel3] trait Cover$Impl extends VerifPrintMacrosDoc { self: cover.type =>

  /** Declares a condition to be covered.
    * At ever clock event, a counter is incremented iff the condition is active
    * and reset is inactive.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * @param cond condition that will be sampled on every clock tick
    * @param message a string describing the cover event
    */
  // Macros currently can't take default arguments, so we need two functions to emulate defaults.
  def apply(cond: Bool, message: String)(implicit sourceInfo: SourceInfo): Cover =
    macro VerifStmtMacrosCompat.cover._applyMacroWithMessage
  def apply(cond: Bool)(implicit sourceInfo: SourceInfo): Cover =
    macro VerifStmtMacrosCompat.cover._applyMacroWithNoMessage
}

private[chisel3] trait Stop$Intf { self: stop.type =>

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a message describing why simulation was stopped
    */
  def apply(message: String)(implicit sourceInfo: SourceInfo): Stop = _applyImpl(message)

  /** Terminate execution, indicating success and printing a message.
    *
    * @param message a printable describing why simulation was stopped
    */
  def apply(message: Printable)(implicit sourceInfo: SourceInfo): Stop = _applyImpl(message)

  /** Terminate execution, indicating success.
    */
  def apply()(implicit sourceInfo: SourceInfo): Stop = _applyImpl()
}
