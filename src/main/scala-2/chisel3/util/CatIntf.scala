// SPDX-License-Identifier: Apache-2.0
package chisel3.util

import chisel3._

import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform

import scala.language.experimental.macros

private[chisel3] trait CatIntf { self: Cat.type =>

  /** Concatenates the argument data elements, in argument order, together. The first argument
    * forms the most significant bits, while the last argument forms the least significant bits.
    */
  def apply[T <: Bits](a: T, r: T*): UInt = macro SourceInfoTransform.arArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Bits](a: T, r: T*)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(a, r: _*)

  /** Concatenates the data elements of the input sequence, in reverse sequence order, together.
    * The first element of the sequence forms the most significant bits, while the last element
    * in the sequence forms the least significant bits.
    *
    * Equivalent to r(0) ## r(1) ## ... ## r(n-1).
    * @note This returns a `0.U` if applied to a zero-element `Vec`.
    */
  def apply[T <: Bits](r: Seq[T]): UInt = macro SourceInfoTransform.rArg

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: Bits](r: Seq[T])(implicit sourceInfo: SourceInfo): UInt = _applyImpl(r)
}
