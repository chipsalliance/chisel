// See LICENSE for license details.

/** Circuit generators for working with one-hot representations.
  */

package chisel3.util

import chisel3._

/** Returns the bit position of the sole high bit of the input bitvector.
  *
  * Inverse operation of [[UIntToOH]].
  *
  * @note assumes exactly one high bit, results undefined otherwise
  */
object OHToUInt {
  def apply(in: Seq[Bool]): UInt = apply(Cat(in.reverse), in.size)
  def apply(in: Vec[Bool]): UInt = apply(in.asUInt, in.size)
  def apply(in: Bits): UInt = apply(in, in.getWidth)

  def apply(in: Bits, width: Int): UInt = {
    if (width <= 2) {
      Log2(in, width)
    } else {
      val mid = 1 << (log2Ceil(width)-1)
      val hi = in(width-1, mid)
      val lo = in(mid-1, 0)
      Cat(hi.orR, apply(hi | lo, mid))
    }
  }
}

/** Returns the bit position of the least-significant high bit of the input bitvector.
  *
  * Multiple bits may be high in the input.
  */
object PriorityEncoder {
  def apply(in: Seq[Bool]): UInt = PriorityMux(in, (0 until in.size).map(_.asUInt))
  def apply(in: Bits): UInt = apply(in.toBools)
}

/** Returns the one hot encoding of the input UInt.
  */
object UIntToOH {
  def apply(in: UInt): UInt = 1.U << in
  def apply(in: UInt, width: Int): UInt = width match {
    case 0 => 0.U(0.W)
    case 1 => 1.U(1.W)
    case _ =>
      val shiftAmountWidth = log2Ceil(width)
      val shiftAmount = in.pad(shiftAmountWidth)(shiftAmountWidth - 1, 0)
      (1.U << shiftAmount)(width - 1, 0)
  }
}

/** Returns a bit vector in which only the least-significant 1 bit in the input vector, if any,
  * is set.
  */
object PriorityEncoderOH {
  private def encode(in: Seq[Bool]): UInt = {
    val outs = Seq.tabulate(in.size)(i => (BigInt(1) << i).asUInt(in.size.W))
    PriorityMux(in :+ true.B, outs :+ 0.U(in.size.W))
  }
  def apply(in: Seq[Bool]): Seq[Bool] = {
    val enc = encode(in)
    Seq.tabulate(in.size)(enc(_))
  }
  def apply(in: Bits): UInt = encode((0 until in.getWidth).map(i => in(i)))
}
