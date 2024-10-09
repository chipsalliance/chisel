// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform

object RegEnable extends RegEnableImpl {

  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    *
    * @example {{{
    * val regWithEnable = RegEnable(nextVal, ena)
    * }}}
    */
  def apply[T <: Data](next: T, enable: Bool): T = macro SourceInfoTransform.nextEnableArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](next: T, enable: Bool)(implicit sourceInfo: SourceInfo): T = _applyImpl(next, enable)

  /** Returns a register with the specified next, update enable gate, and reset initialization.
    *
    * @example {{{
    * val regWithEnableAndReset = RegEnable(nextVal, 0.U, ena)
    * }}}
    */
  def apply[T <: Data](next: T, init: T, enable: Bool): T = macro SourceInfoTransform.nextInitEnableArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    next:   T,
    init:   T,
    enable: Bool
  )(
    implicit sourceInfo: SourceInfo
  ): T = _applyImpl(next, init, enable)
}

object ShiftRegister extends ShiftRegisterImpl {

  /** Returns the n-cycle delayed version of the input signal.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param en enable the shift
    *
    * @example {{{
    * val regDelayTwo = ShiftRegister(nextVal, 2, ena)
    * }}}
    */
  def apply[T <: Data](in: T, n: Int, en: Bool): T = macro SourceInfoTransform.inNEnArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    _applyImpl(in, n, en)

  /** Returns the n-cycle delayed version of the input signal.
    *
    * Enable is assumed to be true.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    *
    * @example {{{
    * val regDelayTwo = ShiftRegister(nextVal, 2)
    * }}}
    */
  def apply[T <: Data](in: T, n: Int): T = macro SourceInfoTransform.inNArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](in: T, n: Int)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(in, n)

  /** Returns the n-cycle delayed version of the input signal with reset initialization.
    *
    * @param in input to delay
    * @param n number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en enable the shift
    *
    * @example {{{
    * val regDelayTwoReset = ShiftRegister(nextVal, 2, 0.U, ena)
    * }}}
    */
  def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): T = macro SourceInfoTransform.inNResetDataEnArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    implicit sourceInfo: SourceInfo
  ): T = _applyImpl(in, n, resetData, en)

  /** Returns the n-cycle delayed version of the input signal (SyncReadMem-based ShiftRegister implementation).
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    * @param en enable the shift
    * @param useDualPortSram dual port or single port SRAM based implementation
    * @param name name of SyncReadMem object
    */
  def mem[T <: Data](in: T, n: Int, en: Bool, useDualPortSram: Boolean, name: Option[String]): T =
    macro SourceInfoTransform.inNEnUseDualPortSramNameArg

  /** @group SourceInfoTransformMacro */
  def do_mem[T <: Data](
    in:              T,
    n:               Int,
    en:              Bool,
    useDualPortSram: Boolean,
    name:            Option[String]
  )(
    implicit sourceInfo: SourceInfo
  ): T = _applyImplMem(in, n, en, useDualPortSram, name)
}

object ShiftRegisters extends ShiftRegistersImpl {

  /** Returns a sequence of delayed input signal registers from 1 to n.
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    * @param en enable the shift
    */
  def apply[T <: Data](in: T, n: Int, en: Bool): Seq[T] = macro SourceInfoTransform.inNEnArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    in: T,
    n:  Int,
    en: Bool
  )(
    implicit sourceInfo: SourceInfo
  ): Seq[T] = _applyImpl(in, n, en)

  /** Returns a sequence of delayed input signal registers from 1 to n.
    *
    * Enable is assumed to be true.
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    */
  def apply[T <: Data](in: T, n: Int): Seq[T] = macro SourceInfoTransform.inNArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](in: T, n: Int)(implicit sourceInfo: SourceInfo): Seq[T] =
    _applyImpl(in, n)

  /** Returns delayed input signal registers with reset initialization from 1 to n.
    *
    * @param in        input to delay
    * @param n         number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en        enable the shift
    */
  def apply[T <: Data](in: T, n: Int, resetData: T, en: Bool): Seq[T] = macro SourceInfoTransform.inNResetDataEnArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    implicit sourceInfo: SourceInfo
  ): Seq[T] = _applyImpl(in, n, resetData, en)
}
