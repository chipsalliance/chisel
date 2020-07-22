// See LICENSE for license details.

package chisel3

import scala.language.experimental.macros
import chisel3.internal.Builder.pushOp
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.internal.firrtl.PrimOp.AsUIntOp

object Clock {
  def apply(): Clock = new Clock
}

// TODO: Document this.
sealed class Clock(private[chisel3] val width: Width = Width(1)) extends Element {
  override def toString: String = s"Clock$bindingToString"

  def cloneType: this.type = Clock().asInstanceOf[this.type]

  private[chisel3] def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass

  override def connect(that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Clock => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  override def litOption: Option[BigInt] = None

  /** Not really supported */
  def toPrintable: Printable = PString("CLOCK")

  /** Returns the contents of the clock wire as a [[Bool]]. */
  final def asBool(): Bool = macro SourceInfoTransform.noArg
  def do_asBool(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Bool = this.asUInt().asBool()

  override def do_asUInt(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  private[chisel3] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that.asBool.asClock
  }
}
