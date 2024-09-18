// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros
import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

object BitPat extends ObjectBitPatImpl {
  implicit class fromUIntToBitPatComparable(x: UInt) extends SourceInfoDoc {

    import scala.language.experimental.macros

    final def ===(that: BitPat): Bool = macro SourceInfoTransform.thatArg
    final def =/=(that: BitPat): Bool = macro SourceInfoTransform.thatArg

    /** @group SourceInfoTransformMacro */
    def do_===(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x

    /** @group SourceInfoTransformMacro */
    def do_=/=(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
  }
}

sealed class BitPat(val value: BigInt, val mask: BigInt, val width: Int) extends BitPatImpl with SourceInfoDoc {
  import chisel3.util.experimental.BitSet
  def terms = Set(this)

  def apply(x:  Int): BitPat = macro SourceInfoTransform.xArg
  def apply(x:  Int, y: Int): BitPat = macro SourceInfoTransform.xyArg
  def ===(that: UInt):   Bool = macro SourceInfoTransform.thatArg
  def =/=(that: UInt):   Bool = macro SourceInfoTransform.thatArg
  def ##(that:  BitPat): BitPat = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int)(implicit sourceInfo: SourceInfo): BitPat = _applyImpl(x)

  /** @group SourceInfoTransformMacro */
  def do_apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): BitPat = _applyImpl(x, y)

  /** @group SourceInfoTransformMacro */
  def do_===(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)

  /** @group SourceInfoTransformMacro */
  def do_##(that: BitPat)(implicit sourceInfo: SourceInfo): BitPat = _impl_##(that)
}
