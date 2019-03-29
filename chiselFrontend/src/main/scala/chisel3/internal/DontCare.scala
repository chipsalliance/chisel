// See LICENSE for license details.

package chisel3.internal

import chisel3.internal.firrtl.{UnknownWidth, Width}
import chisel3.internal.sourceinfo.SourceInfo
//import chisel3.{Bits, CompileOptions, Data, Element, PString, Printable, SpecifiedDirection, UInt, internal}
import chisel3._

/** RHS (source) for Invalidate API.
  * Causes connection logic to emit a DefInvalid when connected to an output port (or wire).
  */
private[chisel3] object DontCare extends Element {
  // This object should be initialized before we execute any user code that refers to it,
  //  otherwise this "Chisel" object will end up on the UserModule's id list.
  // We make it private to chisel3 so it has to be accessed through the package object.

  private[chisel3] override val width: Width = UnknownWidth()

  bind(DontCareBinding(), SpecifiedDirection.Output)
  override def cloneType: this.type = DontCare

  override def toString: String = "DontCare()"

  override def litOption: Option[BigInt] = None

  def toPrintable: Printable = PString("DONTCARE")

  private[chisel3] def connectFromBits(that: Bits)(implicit sourceInfo:  SourceInfo, compileOptions: CompileOptions): Unit = { // scalastyle:ignore line.size.limit
    Builder.error("connectFromBits: DontCare cannot be a connection sink (LHS)")
  }

  def do_asUInt(implicit sourceInfo: internal.sourceinfo.SourceInfo, compileOptions: CompileOptions): UInt = { // scalastyle:ignore line.size.limit
    Builder.error("DontCare does not have a UInt representation")
    0.U
  }
  // DontCare's only match themselves.
  private[chisel3] def typeEquivalent(that: Data): Boolean = that == DontCare
}
