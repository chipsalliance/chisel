// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.language.experimental.macros
import chisel3.experimental.SourceInfo
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.internal.firrtl.PrimOp.AsUIntOp

object Clock {
  def apply(): Clock = new Clock
}

// TODO: Document this.
sealed class Clock(private[chisel3] val width: Width = Width(1)) extends Element {
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

  /** Returns the contents of the clock wire as a [[Bool]]. */
  final def asBool: Bool = macro SourceInfoTransform.noArg

  def do_asBool(implicit sourceInfo: SourceInfo): Bool = this.asUInt.asBool

  override private[chisel3] def _asUIntImpl(first: Boolean)(implicit sourceInfo: SourceInfo): UInt = pushOp(
    DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref)
  )
  private[chisel3] override def connectFromBits(
    that: Bits
  )(
    implicit sourceInfo: SourceInfo
  ): Unit = {
    this := that.asBool.asClock
  }
}
