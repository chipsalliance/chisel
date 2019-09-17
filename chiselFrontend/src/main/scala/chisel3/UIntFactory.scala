// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.{KnownUIntRange, NumericBound, Range, ULit, Width}

// scalastyle:off method.name

// This is currently a factory because both Bits and UInt inherit it.
trait UIntFactory {
  /** Create a UInt type with inferred width. */
  def apply(): UInt = apply(Width())
  /** Create a UInt port with specified width. */
  def apply(width: Width): UInt = new UInt(width)

   /** Create a UInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): UInt = {
    val lit = ULit(value, width)
    val result = new UInt(lit.width)
    // Bind result to being an Literal
    lit.bindLitArg(result)
  }

  /** Create a UInt with the specified range */
  def apply(range: Range): UInt = {
    apply(range.getWidth)
  }
  /** Create a UInt with the specified range */
  def apply(range: (NumericBound[Int], NumericBound[Int])): UInt = {
    apply(KnownUIntRange(range._1, range._2))
  }
}
