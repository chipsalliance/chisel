// See LICENSE for license details.

package chisel3

import chisel3.internal.firrtl.{KnownSIntRange, NumericBound, Range, SLit, Width}

// scalastyle:off method.name

trait SIntFactory {
  /** Create an SInt type with inferred width. */
  def apply(): SInt = apply(Width())
  /** Create a SInt type or port with fixed width. */
  def apply(width: Width): SInt = new SInt(width)

  /** Create a SInt with the specified range */
  def apply(range: Range): SInt = {
    apply(range.getWidth)
  }
  /** Create a SInt with the specified range */
  def apply(range: (NumericBound[Int], NumericBound[Int])): SInt = {
    apply(KnownSIntRange(range._1, range._2))
  }

   /** Create an SInt literal with specified width. */
  protected[chisel3] def Lit(value: BigInt, width: Width): SInt = {
    val lit = SLit(value, width)
    val result = new SInt(lit.width)
    lit.bindLitArg(result)
  }
}
