// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import scala.language.experimental.macros
import chisel3._
import chisel3.experimental.SourceInfo
import chisel3.internal.sourceinfo.SourceInfoTransform
import scala.collection.mutable
import scala.util.hashing.MurmurHash3

private[chisel3] class BaseFomUIntToBitPatComparable(x: UInt) extends SourceInfoDoc {
  import scala.language.experimental.macros

  final def ===(that: BitPat): Bool = macro SourceInfoTransform.thatArg
  final def =/=(that: BitPat): Bool = macro SourceInfoTransform.thatArg

  /** @group SourceInfoTransformMacro */
  def do_===(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that === x

  /** @group SourceInfoTransformMacro */
  def do_=/=(that: BitPat)(implicit sourceInfo: SourceInfo): Bool = that =/= x
}

private[chisel3] trait BitPat$Intf { self: BitPat.type =>
  @deprecated("Use uintToBitPatComparable instead", "Chisel 7.11.0")
  class fromUIntToBitPatComparable(x: UInt) extends BaseFomUIntToBitPatComparable(x)
  @deprecated("Use uintToBitPatComparable instead", "Chisel 7.11.0")
  def fromUIntToBitPatComparable(x: UInt): fromUIntToBitPatComparable = new fromUIntToBitPatComparable(x)
}
private[chisel3] trait BitPatObjIntf { self: BitPat.type =>
  implicit class uintToBitPatComparable(x: UInt) extends BaseFomUIntToBitPatComparable(x)
}

private[chisel3] trait BitPatIntf extends SourceInfoDoc { self: BitPat =>
  import chisel3.util.experimental.BitSet
  def terms = Set(this)

  def apply(x:  Int):         BitPat = macro SourceInfoTransform.xArg
  def apply(x:  Int, y: Int): BitPat = macro SourceInfoTransform.xyArg
  def ===(that: UInt):        Bool = macro SourceInfoTransform.thatArg
  def =/=(that: UInt):        Bool = macro SourceInfoTransform.thatArg
  def ##(that:  BitPat):      BitPat = macro SourceInfoTransform.thatArg

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
