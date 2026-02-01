// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait Reg$Intf { self: Reg.type =>

  /** Construct a [[Reg]] from a type template with no initialization value (reset is ignored).
    * Value will not change unless the [[Reg]] is given a connection.
    * @param t The template from which to construct this wire
    */
  def apply[T <: Data](source: => T)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(source)
}

private[chisel3] trait RegNext$Intf { self: RegNext.type =>

  /** Returns a register ''with an unset width'' connected to the signal `next` and with no reset value. */
  def apply[T <: Data](next: T)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(next)

  /** Returns a register ''with an unset width'' connected to the signal `next` and with the reset value `init`. */
  def apply[T <: Data](next: T, init: T)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(next, init)
}

private[chisel3] trait RegInit$Intf { self: RegInit.type =>

  /** Construct a [[Reg]] from a type template initialized to the specified value on reset
    * @param t The type template used to construct this [[Reg]]
    * @param init The value the [[Reg]] is initialized to on reset
    */
  def apply[T <: Data](t: T, init: T)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(t, init)

  /** Construct a [[Reg]] initialized on reset to the specified value.
    * @param init Initial value that serves as a type template and reset value
    */
  def apply[T <: Data](init: T)(implicit sourceInfo: SourceInfo): T =
    _applyImpl(init)
}
