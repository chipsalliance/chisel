// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform

object RegEnable {

  /** Returns a register with the specified next, update enable gate, and no reset initialization.
    *
    * @example {{{
    * val regWithEnable = RegEnable(nextVal, ena)
    * }}}
    */
  def apply[T <: Data](next: T, enable: Bool): T = macro SourceInfoTransform.nextEnableArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Data](next: T, enable: Bool)(implicit sourceInfo: SourceInfo): T = {
    val r = Reg(chiselTypeOf(next))
    when(enable) { r := next }
    r
  }

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
  ): T = {
    val r = RegInit(init)
    when(enable) { r := next }
    r
  }
}

object ShiftRegister {

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
    _apply_impl(in, n, en)

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
    _apply_impl(in, n)

  private def _apply_impl[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    ShiftRegisters(in, n, en).lastOption.getOrElse(in)

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
  ): T =
    ShiftRegisters(in, n, resetData, en).lastOption.getOrElse(in)

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
  ): T = _apply_impl_mem(in, n, en, useDualPortSram, name)

  private def _apply_impl_mem[T <: Data](
    in:              T,
    n:               Int,
    en:              Bool = true.B,
    useDualPortSram: Boolean = false,
    name:            Option[String] = None
  )(
    implicit sourceInfo: SourceInfo
  ): T = {
    if (n == 0) {
      in
    } else if (n == 1) {
      val out = RegEnable(in, en)
      out
    } else if (useDualPortSram) {
      val mem = SyncReadMem(n, in.cloneType)
      if (name != None) {
        mem.suggestName(name.get)
      }
      val raddr = Counter(en, n)._1
      val out = mem.read(raddr, en)

      val waddr = RegEnable(raddr, (n - 1).U, en)
      when(en) {
        mem.write(waddr, in)
      }
      out
    } else {
      require(n % 2 == 0, "Odd shift register length with single-port SRAMs is not supported")

      val out_sp0 = Wire(in.cloneType)
      out_sp0 := DontCare

      val out_sp1 = Wire(in.cloneType)
      out_sp1 := DontCare

      val mem_sp0 = SyncReadMem(n / 2, in.cloneType)
      val mem_sp1 = SyncReadMem(n / 2, in.cloneType)

      if (name != None) {
        mem_sp0.suggestName(name.get + "_0")
        mem_sp1.suggestName(name.get + "_1")
      }

      val index_counter = Counter(en, n)._1
      val raddr_sp0 = index_counter >> 1.U
      val raddr_sp1 = RegEnable(raddr_sp0, (n / 2 - 1).U, en)

      val wen_sp0 = index_counter(0)
      val wen_sp1 = WireDefault(false.B)
      wen_sp1 := ~wen_sp0

      when(en) {
        val rdwrPort = mem_sp0(raddr_sp0)
        when(wen_sp0) { rdwrPort := in }.otherwise { out_sp0 := rdwrPort }
      }

      when(en) {
        val rdwrPort = mem_sp1(raddr_sp1)
        when(wen_sp1) { rdwrPort := in }.otherwise { out_sp1 := rdwrPort }
      }
      val out = Mux(~wen_sp1, out_sp0, out_sp1)
      out
    }
  }
}

object ShiftRegisters {

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
  ): Seq[T] = _apply_impl(in, n, en)

  private def _apply_impl[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    implicit sourceInfo: SourceInfo
  ): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, en)).drop(1)

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
    _apply_impl(in, n)

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
  ): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, resetData, en)).drop(1)
}
