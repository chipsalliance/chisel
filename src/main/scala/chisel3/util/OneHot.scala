// See LICENSE for license details.

/** Circuit generators for working with one-hot representations.
  */

package chisel3.util

import chisel3._

/** Converts from One Hot Encoding to a UInt indicating which bit is active
  * This is the inverse of [[Chisel.UIntToOH UIntToOH]]*/
object OHToUInt {
  def apply(in: Seq[Bool]): UInt = apply(Vec(in))
  def apply(in: Vec[Bool]): UInt = apply(in.toBits, in.size)
  def apply(in: Bits): UInt = apply(in, in.getWidth)

  def apply(in: Bits, width: Int): UInt = {
    if (width <= 2) {
      Log2(in, width)
    } else {
      val mid = 1 << (log2Up(width)-1)
      val hi = in(width-1, mid)
      val lo = in(mid-1, 0)
      Cat(hi.orR, apply(hi | lo, mid))
    }
  }
}

/** @return the bit position of the trailing 1 in the input vector
  * with the assumption that multiple bits of the input bit vector can be set
  * @example {{{ data_out := PriorityEncoder(data_in) }}}
  */
object PriorityEncoder {
  def apply(in: Seq[Bool]): UInt = PriorityMux(in, (0 until in.size).map(UInt(_)))
  def apply(in: Bits): UInt = apply(in.toBools)
}

/** Returns the one hot encoding of the input UInt.
  */
object UIntToOH
{
  def apply(in: UInt, width: Int = -1): UInt =
    if (width == -1) {
      UInt.Lit(1) << in
    } else {
      (UInt.Lit(1) << in(log2Up(width)-1,0))(width-1,0)
    }
}

/** Returns a bit vector in which only the least-significant 1 bit in
  the input vector, if any, is set.
  */
object PriorityEncoderOH
{
  private def encode(in: Seq[Bool]): UInt = {
    val outs = Seq.tabulate(in.size)(i => UInt(BigInt(1) << i, in.size))
    PriorityMux(in :+ Bool(true), outs :+ UInt(0, in.size))
  }
  def apply(in: Seq[Bool]): Seq[Bool] = {
    val enc = encode(in)
    Seq.tabulate(in.size)(enc(_))
  }
  def apply(in: Bits): UInt = encode((0 until in.getWidth).map(i => in(i)))
}
