// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait RegEnableIntf { self: RegEnable.type =>

  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    *
    * @example {{{
    * val regWithEnable = RegEnable(nextVal, ena)
    * }}}
    */
  def apply[T <: Data](next: T, enable: Bool)(using SourceInfo): T = _applyImpl(next, enable)

  /** Returns a register with the specified next, update enable gate, and reset initialization.
    *
    * @example {{{
    * val regWithEnableAndReset = RegEnable(nextVal, 0.U, ena)
    * }}}
    */
  def apply[T <: Data](
    next:   T,
    init:   T,
    enable: Bool
  )(
    using SourceInfo
  ): T = _applyImpl(next, init, enable)
}

private[chisel3] trait ShiftRegisterIntf { self: ShiftRegister.type =>

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
  def apply[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    using SourceInfo
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
  def apply[T <: Data](in: T, n: Int)(using SourceInfo): T =
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
  def apply[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    using SourceInfo
  ): T = _applyImpl(in, n, resetData, en)

  /** Returns the n-cycle delayed version of the input signal (SyncReadMem-based ShiftRegister implementation).
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    * @param en enable the shift
    * @param useDualPortSram dual port or single port SRAM based implementation
    * @param name name of SyncReadMem object
    */
  def mem[T <: Data](
    in:              T,
    n:               Int,
    en:              Bool,
    useDualPortSram: Boolean,
    name:            Option[String]
  )(
    using SourceInfo
  ): T = _applyImplMem(in, n, en, useDualPortSram, name)
}

private[chisel3] trait ShiftRegistersIntf { self: ShiftRegisters.type =>

  /** Returns a sequence of delayed input signal registers from 1 to n.
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    * @param en enable the shift
    */
  def apply[T <: Data](
    in: T,
    n:  Int,
    en: Bool
  )(
    using SourceInfo
  ): Seq[T] = _applyImpl(in, n, en)

  /** Returns a sequence of delayed input signal registers from 1 to n.
    *
    * Enable is assumed to be true.
    *
    * @param in input to delay
    * @param n  number of cycles to delay
    */
  def apply[T <: Data](in: T, n: Int)(using SourceInfo): Seq[T] =
    _applyImpl(in, n)

  /** Returns delayed input signal registers with reset initialization from 1 to n.
    *
    * @param in        input to delay
    * @param n         number of cycles to delay
    * @param resetData reset value for each register in the shift
    * @param en        enable the shift
    */
  def apply[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    using SourceInfo
  ): Seq[T] = _applyImpl(in, n, resetData, en)
}
