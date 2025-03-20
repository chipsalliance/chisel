// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

private[chisel3] trait BitPat$Intf { self: BitPat.type =>
  implicit class fromUIntToBitPatComparable(x: UInt) {
    def ===(that: BitPat)(using SourceInfo): Bool = that === x
    def =/=(that: BitPat)(using SourceInfo): Bool = that =/= x
  }
}

private[chisel3] trait BitPatIntf extends SourceInfoDoc { self: BitPat =>
  import chisel3.util.experimental.BitSet
  def terms = Set(this)

  def apply(x:  Int)(using SourceInfo):         BitPat = _applyImpl(x)
  def apply(x:  Int, y: Int)(using SourceInfo): BitPat = _applyImpl(x, y)
  def ===(that: UInt)(using SourceInfo):        Bool = _impl_===(that)
  def =/=(that: UInt)(using SourceInfo):        Bool = _impl_=/=(that)
  def ##(that:  BitPat)(using SourceInfo):      BitPat = _impl_##(that)
}
