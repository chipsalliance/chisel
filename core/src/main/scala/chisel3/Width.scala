// SPDX-License-Identifier: Apache-2.0

package chisel3

object Width {
  def apply(x: Int): Width = KnownWidth(x)
  def apply(): Width = UnknownWidth()
}

sealed abstract class Width {
  type W = Int
  def min(that: Width): Width = this.op(that, _ min _)
  def max(that: Width): Width = this.op(that, _ max _)
  def +(that:   Width): Width = this.op(that, _ + _)
  def +(that:   Int):   Width = this.op(this, (a, b) => a + that)
  @deprecated(
    "The width of shift-right now differs by type, use unsignedShiftRight and signedShiftRight",
    "Chisel 7.0.0"
  )
  def shiftRight(that:         Int): Width = this.op(this, (a, b) => 0.max(a - that))
  def unsignedShiftRight(that: Int): Width = this.op(this, (a, b) => 0.max(a - that))
  def signedShiftRight(that:   Int): Width = this.op(this, (a, b) => 1.max(a - that))
  def dynamicShiftLeft(that:   Width): Width =
    this.op(that, (a, b) => a + (1 << b) - 1)

  def known: Boolean
  def get:   W
  protected def op(that: Width, f: (W, W) => W): Width
}

sealed case class UnknownWidth() extends Width {
  def known: Boolean = false
  def get:   Int = None.get
  def op(that: Width, f: (W, W) => W): Width = this
  override def toString: String = ""
}

sealed case class KnownWidth(value: Int) extends Width {
  require(value >= 0, s"Widths must be non-negative, got $value")
  def known: Boolean = true
  def get:   Int = value
  def op(that: Width, f: (W, W) => W): Width = that match {
    case KnownWidth(x) => KnownWidth(f(value, x))
    case _             => that
  }
  override def toString: String = s"<${value.toString}>"
}
