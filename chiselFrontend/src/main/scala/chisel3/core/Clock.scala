// See LICENSE for license details.

package chisel3.core

import chisel3.internal.Builder.{pushOp}
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo._
import chisel3.internal.firrtl.PrimOp.AsUIntOp

object Clock {
  def apply(): Clock = new Clock

  // TODO: move into compatibility package
  def apply(dir: UserDirection)(implicit compileOptions: CompileOptions): Clock = {
    val result = apply()
    dir match {
      case UserDirection.Input => Input(result)
      case UserDirection.Output => Output(result)
    }
  }
}

// TODO: Document this.
sealed class Clock extends Element(Width(1)) {
  def cloneType: this.type = Clock().asInstanceOf[this.type]
  private[chisel3] def toType = "Clock"

  private[core] def typeEquivalent(that: Data): Boolean =
    this.getClass == that.getClass

  override def connect (that: Data)(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): Unit = that match {
    case _: Clock => super.connect(that)(sourceInfo, connectCompileOptions)
    case _ => super.badConnect(that)(sourceInfo)
  }

  /** Not really supported */
  def toPrintable: Printable = PString("CLOCK")

  override def do_asUInt(implicit sourceInfo: SourceInfo, connectCompileOptions: CompileOptions): UInt = pushOp(DefPrim(sourceInfo, UInt(this.width), AsUIntOp, ref))
  private[core] override def connectFromBits(that: Bits)(implicit sourceInfo: SourceInfo,
      compileOptions: CompileOptions): Unit = {
    this := that
  }
}
