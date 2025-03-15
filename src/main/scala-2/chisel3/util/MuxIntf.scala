// SPDX-License-Identifier: Apache-2.0

/** Mux circuit generators.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.{MuxLookupTransform, SourceInfoTransform}
import scala.language.experimental.macros

private[chisel3] trait Mux1H$Intf { self: Mux1H.type =>

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T]): T = macro SourceInfoTransform.selInArg
  def do_apply[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](in:    Iterable[(Bool, T)]):                                  T = macro SourceInfoTransform.inArg
  def do_apply[T <: Data](in: Iterable[(Bool, T)])(implicit sourceInfo: SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: UInt, in: Seq[T]): T = macro SourceInfoTransform.selInArg
  def do_apply[T <: Data](sel: UInt, in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply(sel:    UInt, in: UInt):                                  Bool = macro SourceInfoTransform.selInArg
  def do_apply(sel: UInt, in: UInt)(implicit sourceInfo: SourceInfo): Bool = _applyImpl(sel, in)

}

private[chisel3] trait PriorityMux$Intf { self: PriorityMux.type =>

  def apply[T <: Data](in:    Seq[(Bool, T)]):                                  T = macro SourceInfoTransform.inArg
  def do_apply[T <: Data](in: Seq[(Bool, T)])(implicit sourceInfo: SourceInfo): T = _applyImpl(in)

  def apply[T <: Data](sel: Seq[Bool], in: Seq[T]): T = macro SourceInfoTransform.selInArg
  def do_apply[T <: Data](sel: Seq[Bool], in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)

  def apply[T <: Data](sel: Bits, in: Seq[T]): T = macro SourceInfoTransform.selInArg
  def do_apply[T <: Data](sel: Bits, in: Seq[T])(implicit sourceInfo: SourceInfo): T = _applyImpl(sel, in)
}

private[chisel3] trait MuxLookup$Intf extends SourceInfoDoc { self: MuxLookup.type =>

  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[T <: Data](key: UInt, default: T)(mapping: Seq[(UInt, T)]): T =
    macro MuxLookupTransform.applyCurried[UInt, T]

  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[S <: EnumType, T <: Data](key: S, default: T)(mapping: Seq[(S, T)]): T =
    macro MuxLookupTransform.applyEnum[S, T]

  /** @group SourceInfoTransformMacro */
  def do_applyEnum[S <: EnumType, T <: Data](
    key:     S,
    default: T,
    mapping: Seq[(S, T)]
  )(
    implicit sourceinfo: SourceInfo
  ): T = _applyEnumImpl(key, default, mapping)

  /** @group SourceInfoTransformMacro */
  def do_apply[S <: UInt, T <: Data](key: S, default: T, mapping: Seq[(S, T)])(implicit sourceinfo: SourceInfo): T =
    _applyImpl(key, default, mapping)
}
