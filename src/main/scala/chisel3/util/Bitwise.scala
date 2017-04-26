// See LICENSE for license details.

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.core.SeqUtils

/** Creates repetitions of each bit of the input in order.
  *
  * @example {{{
  * FillInterleaved(2, "b1 0 0 0".U)  // equivalent to "b11 00 00 00".U
  * FillInterleaved(2, "b1 0 0 1".U)  // equivalent to "b11 00 00 11".U
  * FillInterleaved(2, myUIntWire)  // dynamic interleaved fill
  *
  * FillInterleaved(2, Seq(true.B, false.B, false.B, false.B))  // equivalent to "b11 00 00 00".U
  * FillInterleaved(2, Seq(true.B, false.B, false.B, true.B))  // equivalent to "b11 00 00 11".U
  * }}}
  */
object FillInterleaved {
  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: UInt): UInt = apply(n, in.toBools)

  /** Creates n repetitions of each bit of x in order.
    *
    * Output data-equivalent to in(size(in)-1) (n times) ## ... ## in(1) (n times) ## in(0) (n times)
    */
  def apply(n: Int, in: Seq[Bool]): UInt = Cat(in.map(Fill(n, _)).reverse)
}

/** Returns the number of bits set (value is 1 or true) in the input signal.
  *
  * @example {{{
  * PopCount(Seq(true.B, false.B, true.B, true.B))  // evaluates to 3.U
  * PopCount(Seq(false.B, false.B, true.B, false.B))  // evaluates to 1.U
  *
  * PopCount("b1011".U)  // evaluates to 3.U
  * PopCount("b0010".U)  // evaluates to 1.U
  * PopCount(myUIntWire)  // dynamic count
  * }}}
  */
object PopCount {
  def apply(in: Iterable[Bool]): UInt = SeqUtils.count(in.toSeq)

  def apply(in: Bits): UInt = apply((0 until in.getWidth).map(in(_)))
}

/** Create repetitions of the input using a tree fanout topology.
  *
  * @example {{{
  * Fill(2, "b1000".U)  // equivalent to "b1000 1000".U
  * Fill(2, "b1001".U)  // equivalent to "b1001 1001".U
  * Fill(2, myUIntWire)  // dynamic fill
  * }}}
  */
object Fill {
  /** Create n repetitions of x using a tree fanout topology.
    *
    * Output data-equivalent to x ## x ## ... ## x (n repetitions).
    */
  def apply(n: Int, x: UInt): UInt = {
    n match {
      case _ if n < 0 => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
      case 0 => UInt(0.W)
      case 1 => x
      case _ if x.isWidthKnown && x.getWidth == 1 =>
        Mux(x.toBool, ((BigInt(1) << n) - 1).asUInt(n.W), 0.U(n.W))
      case _ =>
        val nBits = log2Ceil(n + 1)
        val p2 = Array.ofDim[UInt](nBits)
        p2(0) = x
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i-1), p2(i-1))
        Cat((0 until nBits).filter(i => (n & (1 << i)) != 0).map(p2(_)))
    }
  }
}

/** Returns the input in bit-reversed order. Useful for little/big-endian conversion.
  *
  * @example {{{
  * Reverse("b1101".U)  // equivalent to "b1011".U
  * Reverse("b1101".U(8.W))  // equivalent to "b10110000".U
  * Reverse(myUIntWire)  // dynamic reverse
  * }}}
  */
object Reverse {
  private def doit(in: UInt, length: Int): UInt = length match {
    case _ if length < 0 => throw new IllegalArgumentException(s"length (=$length) must be nonnegative integer.")
    case _ if length <= 1 => in
    case _ if isPow2(length) && length >= 8 && length <= 64 =>
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
    case _ =>
      val half = (1 << log2Ceil(length))/2
      Cat(doit(in(half-1,0), half), doit(in(length-1,half), length-half))
  }

  def apply(in: UInt): UInt = doit(in, in.getWidth)
}
