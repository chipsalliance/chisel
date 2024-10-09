// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait RegEnableImpl {

  protected def _applyImpl[T <: Data](next: T, enable: Bool)(implicit sourceInfo: SourceInfo): T = {
    val r = Reg(chiselTypeOf(next))
    when(enable) { r := next }
    r
  }

  protected def _applyImpl[T <: Data](
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

private[chisel3] trait ShiftRegisterImpl {

  protected def _applyImpl[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    ShiftRegisters(in, n, en).lastOption.getOrElse(in)

  protected def _applyImpl[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    implicit sourceInfo: SourceInfo
  ): T =
    ShiftRegisters(in, n, resetData, en).lastOption.getOrElse(in)

  protected def _applyImplMem[T <: Data](
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

private[chisel3] trait ShiftRegistersImpl {

  protected def _applyImpl[T <: Data](
    in: T,
    n:  Int,
    en: Bool = true.B
  )(
    implicit sourceInfo: SourceInfo
  ): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, en)).drop(1)

  protected def _applyImpl[T <: Data](
    in:        T,
    n:         Int,
    resetData: T,
    en:        Bool
  )(
    implicit sourceInfo: SourceInfo
  ): Seq[T] =
    Seq.iterate(in, n + 1)(util.RegEnable(_, resetData, en)).drop(1)
}
