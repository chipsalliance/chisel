// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl.ir._
import chisel3.internal.firrtl.ir.PrimOp.AsUIntOp

// TODO: Document this.
private[chisel3] trait ClockImpl extends Element {
  private[chisel3] val width: Width = Width(1)

  override def toString: String = stringAccessor("Clock")

  def cloneType: this.type = Clock().asInstanceOf[this.type]

  override def connect(that: Data)(implicit sourceInfo: SourceInfo): Unit =
    that match {
      case _: Clock | DontCare => super.connect(that)(sourceInfo)
      case _ => super.badConnect(that)(sourceInfo)
    }

  override def litOption: Option[BigInt] = None

  /** Not really supported */
  def toPrintable: Printable = PString("CLOCK")

  protected def _asBoolImpl(implicit sourceInfo: SourceInfo): Bool = this.asUInt.asBool

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )

  override private[chisel3] def _fromUInt(that: UInt)(implicit sourceInfo: SourceInfo): Data = that.asBool.asClock
}
