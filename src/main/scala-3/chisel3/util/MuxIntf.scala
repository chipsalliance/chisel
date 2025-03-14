// SPDX-License-Identifier: Apache-2.0

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait Mux1H$Intf { self: Mux1H.type =>

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T])(using SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](in: Iterable[(Bool, T)])(using SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: UInt, in: Seq[T])(using SourceInfo): T = _applyImpl(sel, in)

  def apply(sel: UInt, in: UInt)(using SourceInfo): Bool = _applyImpl(sel, in)
}

private[chisel3] trait PriorityMux$Intf { self: PriorityMux.type =>

  def apply[T <: Data](in: Seq[(Bool, T)])(using SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T])(using SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](sel: Bits, in: Seq[T])(using SourceInfo): T = _applyImpl(sel, in)
}

private[chisel3] trait MuxLookup$Intf { self: MuxLookup.type =>

  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def applyEnum[S <: EnumType, T <: Data](
    key:     S,
    default: T,
    mapping: Seq[(S, T)]
  )(
    using SourceInfo
  ): T = _applyEnumImpl(key, default, mapping)

  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[S <: UInt, T <: Data](key: S, default: T, mapping: Seq[(S, T)])(using SourceInfo): T =
    _applyImpl(key, default, mapping)
}
