// SPDX-License-Identifier: Apache-2.0

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

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
