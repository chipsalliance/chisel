// SPDX-License-Identifier: Apache-2.0

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

/** Builds a Mux tree out of the input signal vector using a one hot encoded
  * select signal. Returns the output of the Mux tree.
  *
  * @example {{{
  * val hotValue = chisel3.util.Mux1H(Seq(
  *  io.selector(0) -> 2.U,
  *  io.selector(1) -> 4.U,
  *  io.selector(2) -> 8.U,
  *  io.selector(4) -> 11.U,
  * ))
  * }}}
  *
  * @note results unspecified unless exactly one select signal is high
  */
object Mux1H extends Mux1HImpl {

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](in: Iterable[(Bool, T)])(implicit sourceInfo: SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: UInt, in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply(sel: UInt, in: UInt)(implicit sourceInfo: SourceInfo): Bool = _applyImpl(sel, in)
}

/** Builds a Mux tree under the assumption that multiple select signals
  * can be enabled. Priority is given to the first select signal.
  *
  * @example {{{
  * val hotValue = chisel3.util.PriorityMux(Seq(
  *  io.selector(0) -> 2.U,
  *  io.selector(1) -> 4.U,
  *  io.selector(2) -> 8.U,
  *  io.selector(4) -> 11.U,
  * ))
  * }}}
  * Returns the output of the Mux tree.
  */
object PriorityMux extends PriorityMuxImpl {

  def apply[T <: Data](in: Seq[(Bool, T)])(implicit sourceInfo: SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](sel: Bits, in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)
}

/** Creates a cascade of n Muxs to search for a key value. The Selector may be a UInt or an EnumType.
  *
  * @example {{{
  * MuxLookup(idx, default)(Seq(0.U -> a, 1.U -> b))
  * MuxLookup(myEnum, default)(Seq(MyEnum.a -> 1.U, MyEnum.b -> 2.U, MyEnum.c -> 3.U))
  * }}}
  */
object MuxLookup extends MuxLookupImpl {

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
    implicit sourceinfo: SourceInfo
  ): T = _applyEnumImpl(key, default, mapping)

  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[S <: UInt, T <: Data](key: S, default: T, mapping: Seq[(S, T)])(implicit sourceinfo: SourceInfo): T =
    _applyImpl(key, default, mapping)
}
