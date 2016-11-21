// See LICENSE for license details.

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.core.SeqUtils

object FillInterleaved {
  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    * For example, FillInterleaved(2, "b1000") === UInt("b11 00 00 00")
    */
  def apply(n: Int, in: UInt): UInt = apply(n, in.toBools)

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: Seq[Bool]): UInt = Cat(in.map(Fill(n, _)).reverse)
}

/** Returns the number of bits set (i.e value is 1) in the input signal.
  */
object PopCount
{
  def apply(in: Iterable[Bool]): UInt = SeqUtils.count(in.toSeq)
  def apply(in: Bits): UInt = apply((0 until in.getWidth).map(in(_)))
}

object Fill {
  /** Create n repetitions of x using a tree fanout topology.
    *
    * Output data-equivalent to x ## x ## ... ## x (n repetitions).
    */
  def apply(n: Int, x: UInt): UInt = {
    n match {
      case 0 => UInt(0.W)
      case 1 => x
      case _ if x.isWidthKnown && x.getWidth == 1 =>
        Mux(x.toBool, ((BigInt(1) << n) - 1).asUInt(n.W), 0.U(n.W))
      case _ if n > 1 =>
        val p2 = Array.ofDim[UInt](log2Up(n + 1))
        p2(0) = x
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i-1), p2(i-1))
        Cat((0 until log2Up(n + 1)).filter(i => (n & (1 << i)) != 0).map(p2(_)))
      case _ => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
    }
  }
}

object Reverse {
  private def doit(in: UInt, length: Int): UInt = {
    if (length == 1) {
      in
    } else if (isPow2(length) && length >= 8 && length <= 64) {
      // This esoterica improves simulation performance
      var res = in
      var shift = length >> 1
      var mask = ((BigInt(1) << length) - 1).asUInt(length.W)
      do {
        mask = mask ^ (mask(length-shift-1,0) << shift)
        res = ((res >> shift) & mask) | ((res(length-shift-1,0) << shift) & ~mask)
        shift = shift >> 1
      } while (shift > 0)
      res
    } else {
      val half = (1 << log2Up(length))/2
      Cat(doit(in(half-1,0), half), doit(in(length-1,half), length-half))
    }
  }
  /** Returns the input in bit-reversed order. Useful for little/big-endian conversion.
    */
  def apply(in: UInt): UInt = doit(in, in.getWidth)
}
