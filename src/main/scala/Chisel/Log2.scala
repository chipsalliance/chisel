package Chisel

// TODO: these operators should be backend nodes so their width can be
// inferred, rather than using getWidth.  also, C++ perf would improve.

object Log2 {
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) UInt(0)
    else if (width == 2) x(1)
    else Mux(x(width-1), UInt(width-1), apply(x, width-1))
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)
}

object OHToUInt {
  def apply(in: Seq[Bool]): UInt = apply(Vec(in))
  def apply(in: Vec[Bool]): UInt = apply(in.toBits, in.size)
  def apply(in: Bits): UInt = apply(in, in.getWidth)

  def apply(in: Bits, width: Int): UInt = {
    if (width <= 2) Log2(in, width)
    else {
      val mid = 1 << (log2Up(width)-1)
      val hi = in(width-1, mid)
      val lo = in(mid-1, 0)
      Cat(hi.orR, apply(hi | lo, mid))
    }
  }
}

object PriorityEncoder {
  def apply(in: Iterable[Bool]): UInt = PriorityMux(in, (0 until in.size).map(UInt(_)))
  def apply(in: Bits): UInt = apply(in.toBools)
}
