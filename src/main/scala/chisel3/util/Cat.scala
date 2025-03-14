// SPDX-License-Identifier: Apache-2.0
package chisel3.util

import chisel3._

import chisel3.experimental.SourceInfo

/** Concatenates elements of the input, in order, together.
  *
  * @example {{{
  * Cat("b101".U, "b11".U)  // equivalent to "b101 11".U
  * Cat(myUIntWire0, myUIntWire1)
  *
  * Cat(Seq("b101".U, "b11".U))  // equivalent to "b101 11".U
  * Cat(mySeqOfBits)
  * }}}
  */
object Cat extends CatIntf {

  protected def _applyImpl[T <: Bits](a: T, r: T*)(implicit sourceInfo: SourceInfo): UInt =
    _applyImpl(a :: r.toList)

  protected def _applyImpl[T <: Bits](r: Seq[T])(implicit sourceInfo: SourceInfo): UInt =
    SeqUtils.asUInt(r.reverse)
}
