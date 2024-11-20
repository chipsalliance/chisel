// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

object BitPat extends ObjectBitPatImpl {
  implicit class fromUIntToBitPatComparable(x: UInt) {
    def ===(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x
    def =/=(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
  }
}

sealed class BitPat(val value: BigInt, val mask: BigInt, val width: Int) extends BitPatImpl with SourceInfoDoc {
  import chisel3.util.experimental.BitSet
  def terms = Set(this)

  def apply(x: Int)(implicit sourceInfo: SourceInfo): BitPat = _applyImpl(x)
  def apply(x: Int, y: Int)(implicit sourceInfo: SourceInfo): BitPat = _applyImpl(x, y)
  def ===(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_===(that)
  def =/=(that: UInt)(implicit sourceInfo: SourceInfo): Bool = _impl_=/=(that)
  def ##(that: BitPat)(implicit sourceInfo: SourceInfo): BitPat = _impl_##(that)
}
