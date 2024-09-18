// SPDX-License-Identifier: Apache-2.0

/** Miscellaneous circuit generators operating on bits.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.SourceInfo

private[chisel3] trait FillInterleavedImpl {

  protected def _applyImpl(n: Int, in: UInt)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(n, in.asBools)

  protected def _applyImpl(n: Int, in: Seq[Bool])(implicit sourceInfo: SourceInfo): UInt =
    Cat(in.map(Fill(n, _)).reverse)
}

private[chisel3] trait PopCountImpl {

  protected def _applyImpl(in: Iterable[Bool])(implicit sourceInfo: SourceInfo): UInt = SeqUtils.count(in.toSeq)

  protected def _applyImpl(in: Bits)(implicit sourceInfo: SourceInfo): UInt = _applyImpl(
    (0 until in.getWidth).map(in(_))
  )
}

private[chisel3] trait FillImpl {

  protected def _applyImpl(n: Int, x: UInt)(implicit sourceInfo: SourceInfo): UInt = {
    n match {
      case _ if n < 0 => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
      case 0          => UInt(0.W)
      case 1          => x
      case _ if x.isWidthKnown && x.getWidth == 1 =>
        Mux(x.asBool, ((BigInt(1) << n) - 1).asUInt(n.W), 0.U(n.W))
      case _ =>
        val nBits = log2Ceil(n + 1)
        val p2 = Array.ofDim[UInt](nBits)
        p2(0) = x
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i - 1), p2(i - 1))
        Cat((0 until nBits).filter(i => (n & (1 << i)) != 0).map(p2(_)))
    }
  }
}

private[chisel3] trait ReverseImpl {

  private def doit(in: UInt, length: Int)(implicit sourceInfo: SourceInfo): UInt =
    length match {
      case _ if length < 0                                    => throw new IllegalArgumentException(s"length (=$length) must be nonnegative integer.")
      case _ if length <= 1                                   => in
      case _ if isPow2(length) && length >= 8 && length <= 64 =>
        // This esoterica improves simulation performance
        var res = in
        var shift = length >> 1
        var mask = ((BigInt(1) << length) - 1).asUInt(length.W)
        do {
          mask = mask ^ (mask(length - shift - 1, 0) << shift)
          res = ((res >> shift) & mask) | ((res(length - shift - 1, 0) << shift) & ~mask)
          shift = shift >> 1
        } while (shift > 0)
        res
      case _ =>
        val half = (1 << log2Ceil(length)) / 2
        Cat(doit(in(half - 1, 0), half), doit(in(length - 1, half), length - half))
    }

  protected def _applyImpl(in: UInt)(implicit sourceInfo: SourceInfo): UInt = doit(in, in.getWidth)
}
