// See LICENSE for license details.

/** Miscellaneous circuit generators operating on bits.
  */

package Chisel

object FillInterleaved
{
  def apply(n: Int, in: UInt): UInt = apply(n, in.toBools)
  def apply(n: Int, in: Seq[Bool]): UInt = Vec(in.map(Fill(n, _))).toBits
}

/** Returns the number of bits set (i.e value is 1) in the input signal.
  */
object PopCount
{
  def apply(in: Iterable[Bool]): UInt = {
    if (in.size == 0) {
      UInt(0)
    } else if (in.size == 1) {
      in.head
    } else {
      apply(in.slice(0, in.size/2)) + Cat(UInt(0), apply(in.slice(in.size/2, in.size)))
    }
  }
  def apply(in: Bits): UInt = apply((0 until in.getWidth).map(in(_)))
}

/** Fill fans out a UInt to multiple copies */
object Fill {
  /** Fan out x n times */
  def apply(n: Int, x: UInt): UInt = {
    n match {
      case 0 => UInt(width=0)
      case 1 => x
      case y if n > 1 =>
        val p2 = Array.ofDim[UInt](log2Up(n + 1))
        p2(0) = x
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i-1), p2(i-1))
        Cat((0 until log2Up(y + 1)).filter(i => (y & (1 << i)) != 0).map(p2(_)))
      case _ => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
    }
  }
  /** Fan out x n times */
  def apply(n: Int, x: Bool): UInt =
    if (n > 1) {
      UInt(0,n) - x
    } else {
      apply(n, x: UInt)
    }
}

/** Litte/big bit endian convertion: reverse the order of the bits in a UInt.
*/
object Reverse
{
  private def doit(in: UInt, length: Int): UInt = {
    if (length == 1) {
      in
    } else if (isPow2(length) && length >= 8 && length <= 64) {
      // This esoterica improves simulation performance
      var res = in
      var shift = length >> 1
      var mask = UInt((BigInt(1) << length) - 1, length)
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
  def apply(in: UInt): UInt = doit(in, in.getWidth)
}
