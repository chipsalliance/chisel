package Chisel
import Builder._
import scala.math._

object log2Up
{
  def apply(in: Int): Int = if(in == 1) 1 else ceil(log(in)/log(2)).toInt
}

object log2Ceil
{
  def apply(in: Int): Int = ceil(log(in)/log(2)).toInt
}


object log2Down
{
  def apply(x : Int): Int = if (x == 1) 1 else floor(log(x)/log(2.0)).toInt
}

object log2Floor
{
  def apply(x : Int): Int = floor(log(x)/log(2.0)).toInt
}


object isPow2
{
  def apply(in: Int): Boolean = in > 0 && ((in & (in-1)) == 0)
}

object Log2 {
  def apply(x: UInt, width: Int): UInt = {
    if (width < 2) UInt(0)
    else Mux(x(width-1), UInt(width-1), apply(x, width-1))
  }

  // TODO: infer the width in the backend
  def apply(x: UInt): UInt = apply(x, x.getWidth)
}

object FillInterleaved
{
  def apply(n: Int, in: Bits): Bits = apply(n, in.toBools)
  def apply(n: Int, in: Seq[Bool]): Bits = Vec(in.map(Fill(n, _))).toBits
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

object RegNext {

  def apply[T <: Data](next: T): T = Reg[T](next, next, null.asInstanceOf[T])

  def apply[T <: Data](next: T, init: T): T = Reg[T](next, next, init)

}

object RegInit {

  def apply[T <: Data](init: T): T = Reg[T](init, null.asInstanceOf[T], init)

}

object RegEnable
{
  def apply[T <: Data](updateData: T, enable: Bool) = {
    val r = Reg(updateData)
    when (enable) { r := updateData }
    r
  }
  def apply[T <: Data](updateData: T, resetData: T, enable: Bool) = {
    val r = RegInit(resetData)
    when (enable) { r := updateData }
    r
  }
}

/** Builds a Mux tree out of the input signal vector using a one hot encoded
  select signal. Returns the output of the Mux tree.
  */
object Mux1H
{
  def apply[T <: Data](sel: Iterable[Bool], in: Iterable[T]): T = {
    if (in.tail.isEmpty) in.head
    else {
      val masked = (sel, in).zipped map ((s, i) => Mux(s, i.toBits, Bits(0)))
      in.head.fromBits(masked.reduceLeft(_|_))
    }
  }
  def apply[T <: Data](in: Iterable[(Bool, T)]): T = {
    val (sel, data) = in.unzip
    apply(sel, data)
  }
  def apply[T <: Data](sel: Bits, in: Iterable[T]): T =
    apply((0 until in.size).map(sel(_)), in)
  def apply(sel: Bits, in: Bits): Bool = (sel & in).orR
}

/** Builds a Mux tree under the assumption that multiple select signals
  can be enabled. Priority is given to the first select signal.

  Returns the output of the Mux tree.
  */
object PriorityMux
{
  def apply[T <: Bits](in: Iterable[(Bool, T)]): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      Mux(in.head._1, in.head._2, apply(in.tail))
    }
  }
  def apply[T <: Bits](sel: Iterable[Bool], in: Iterable[T]): T = apply(sel zip in)
  def apply[T <: Bits](sel: Bits, in: Iterable[T]): T = apply((0 until in.size).map(sel(_)), in)
}

object unless {
  def apply(c: Bool)(block: => Unit) {
    when (!c) { block }
  }
}

object switch {
  def apply(c: Bits)(block: => Unit) {
    switchKeys.push(c)
    block
    switchKeys.pop()
  }
}

object is {
  def apply(v: Bits)(block: => Unit): Unit =
    apply(Seq(v))(block)
  def apply(v: Bits, vr: Bits*)(block: => Unit): Unit =
    apply(v :: vr.toList)(block)
  def apply(v: Iterable[Bits])(block: => Unit): Unit = {
    val keys = switchKeys
    if (keys.isEmpty) ChiselError.error("The 'is' keyword may not be used outside of a switch.")
    else if (!v.isEmpty) when (v.map(_ === keys.top).reduce(_||_)) { block }
  }
}

object MuxLookup {
  def apply[S <: UInt, T <: Bits] (key: S, default: T, mapping: Seq[(S, T)]): T = {
    var res = default;
    for ((k, v) <- mapping.reverse)
      res = Mux(k === key, v, res);
    res
  }

}

object Fill {
  def apply(n: Int, x: Bool): UInt = n match {
    case 0 => UInt(width=0)
    case 1 => x
    case x if n > 1 => UInt(0,n) - UInt(x)
    case _ => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
  }
  def apply(n: Int, y: UInt): UInt = {
    n match {
      case 0 => UInt(width=0)
      case 1 => y
      case x if n > 1 =>
        val p2 = Array.ofDim[UInt](log2Up(n+1))
        p2(0) = y
        for (i <- 1 until p2.length)
          p2(i) = Cat(p2(i-1), p2(i-1))
        Cat((0 until log2Up(x+1)).filter(i => (x & (1 << i)) != 0).map(p2(_)))
      case _ => throw new IllegalArgumentException(s"n (=$n) must be nonnegative integer.")
    }
  }
}

object MuxCase {
  def apply[T <: Bits] (default: T, mapping: Seq[(Bool, T)]): T = {
    var res = default;
    for ((t, v) <- mapping.reverse){
      res = Mux(t, v, res);
    }
    res
  }
}

object ListLookup {
  def apply[T <: Data](addr: UInt, default: List[T], mapping: Array[(BitPat, List[T])]): List[T] = {
    val map = mapping.map(m => (m._1 === addr, m._2))
    default.zipWithIndex map { case (d, i) =>
      map.foldRight(d)((m, n) => Mux(m._1, m._2(i), n))
    }
  }
}

object Lookup {
  def apply[T <: Bits](addr: UInt, default: T, mapping: Seq[(BitPat, T)]): T =
    ListLookup(addr, List(default), mapping.map(m => (m._1, List(m._2))).toArray).head
}

/** Litte/big bit endian convertion: reverse the order of the bits in a UInt.
*/
object Reverse
{
  private def doit(in: UInt, length: Int): UInt = {
    if (length == 1) {
      in
    } else if (isPow2(length) && length >= 8 && length <= 64) {
      // Do it in logarithmic time to speed up C++.  Neutral for real HW.
      var res = in
      var shift = length >> 1
      var mask = UInt((BigInt(1) << length) - 1, length)
      do {
        mask = mask ^ (mask(length-shift-1,0) << UInt(shift))
        res = ((res >> UInt(shift)) & mask) | (res(length-shift-1,0) << UInt(shift) & ~mask)
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

/** Returns the n-cycle delayed version of the input signal.
  */
object ShiftRegister
{
  def apply[T <: Data](in: T, n: Int, en: Bool = Bool(true)): T =
  {
    // The order of tests reflects the expected use cases.
    if (n == 1) {
      RegEnable(in, en)
    } else if (n != 0) {
      RegNext(apply(in, n-1, en))
    } else {
      in
    }
  }
}

/** Returns the one hot encoding of the input UInt.
  */
object UIntToOH
{
  def apply(in: UInt, width: Int = -1): UInt =
    if (width == -1) UInt(1) << in
    else (UInt(1) << in(log2Up(width)-1,0))(width-1,0)
}

class Counter(val n: Int) {
  val value = if (n == 1) UInt(0) else Reg(init=UInt(0, log2Up(n)))
  def inc(): Bool = {
    if (n == 1) Bool(true)
    else {
      val wrap = value === UInt(n-1)
      value := Mux(Bool(!isPow2(n)) && wrap, UInt(0), value + UInt(1))
      wrap
    }
  }
}

object Counter
{
  def apply(n: Int): Counter = new Counter(n)
  def apply(cond: Bool, n: Int): (UInt, Bool) = {
    val c = new Counter(n)
    var wrap: Bool = null
    when (cond) { wrap = c.inc() }
    (c.value, cond && wrap)
  }
}

class ValidIO[+T <: Data](gen: T) extends Bundle
{
  val valid = Bool(OUTPUT)
  val bits = gen.cloneType.asOutput
  def fire(dummy: Int = 0): Bool = valid
  override def cloneType: this.type = new ValidIO(gen).asInstanceOf[this.type]
}

/** Adds a valid protocol to any interface. The standard used is
  that the consumer uses the flipped interface.
*/
object Valid {
  def apply[T <: Data](gen: T): ValidIO[T] = new ValidIO(gen)
}

class DecoupledIO[+T <: Data](gen: T) extends Bundle
{
  val ready = Bool(INPUT)
  val valid = Bool(OUTPUT)
  val bits  = gen.cloneType.asOutput
  def fire(dummy: Int = 0): Bool = ready && valid
  override def cloneType: this.type = new DecoupledIO(gen).asInstanceOf[this.type]
}

/** Adds a ready-valid handshaking protocol to any interface.
  The standard used is that the consumer uses the flipped
  interface.
  */
object Decoupled {
  def apply[T <: Data](gen: T): DecoupledIO[T] = new DecoupledIO(gen)
}

class EnqIO[T <: Data](gen: T) extends DecoupledIO(gen)
{
  def enq(dat: T): T = { valid := Bool(true); bits := dat; dat }
  valid := Bool(false);
  for (io <- bits.flatten)
    io := UInt(0)
  override def cloneType: this.type = { new EnqIO(gen).asInstanceOf[this.type]; }
}

class DeqIO[T <: Data](gen: T) extends DecoupledIO(gen)
{
  flip()
  ready := Bool(false);
  def deq(b: Boolean = false): T = { ready := Bool(true); bits }
  override def cloneType: this.type = { new DeqIO(gen).asInstanceOf[this.type]; }
}


class DecoupledIOC[+T <: Data](gen: T) extends Bundle
{
  val ready = Bool(INPUT)
  val valid = Bool(OUTPUT)
  val bits  = gen.cloneType.asOutput
}

class QueueIO[T <: Data](gen: T, entries: Int) extends Bundle
{
  val enq   = Decoupled(gen.cloneType).flip
  val deq   = Decoupled(gen.cloneType)
  val count = UInt(OUTPUT, log2Up(entries + 1))
}

class Queue[T <: Data](gen: T, val entries: Int, pipe: Boolean = false, flow: Boolean = false, _reset: Bool = null) extends Module(_reset=_reset)
{
  val io = new QueueIO(gen, entries)

  val ram = Mem(gen, entries)
  val enq_ptr = Counter(entries)
  val deq_ptr = Counter(entries)
  val maybe_full = Reg(init=Bool(false))

  val ptr_match = enq_ptr.value === deq_ptr.value
  val empty = ptr_match && !maybe_full
  val full = ptr_match && maybe_full
  val maybe_flow = Bool(flow) && empty
  val do_flow = maybe_flow && io.deq.ready

  val do_enq = io.enq.ready && io.enq.valid && !do_flow
  val do_deq = io.deq.ready && io.deq.valid && !do_flow
  when (do_enq) {
    ram(enq_ptr.value) := io.enq.bits
    enq_ptr.inc()
  }
  when (do_deq) {
    deq_ptr.inc()
  }
  when (do_enq != do_deq) {
    maybe_full := do_enq
  }

  io.deq.valid := !empty || Bool(flow) && io.enq.valid
  io.enq.ready := !full || Bool(pipe) && io.deq.ready
  io.deq.bits := Mux(maybe_flow, io.enq.bits, ram(deq_ptr.value))

  val ptr_diff = enq_ptr.value - deq_ptr.value
  if (isPow2(entries)) {
    io.count := Cat(maybe_full && ptr_match, ptr_diff)
  } else {
    io.count := Mux(ptr_match, Mux(maybe_full, UInt(entries), UInt(0)), Mux(deq_ptr.value > enq_ptr.value, UInt(entries) + ptr_diff, ptr_diff))
  }
}

/** Generic hardware queue. Required parameter entries controls
  the depth of the queues. The width of the queue is determined
  from the inputs.

  Example usage:
    val q = new Queue(UInt(), 16)
    q.io.enq <> producer.io.out
    consumer.io.in <> q.io.deq
  */
object Queue
{
  def apply[T <: Data](enq: DecoupledIO[T], entries: Int = 2, pipe: Boolean = false): DecoupledIO[T]  = {
    val q = Module(new Queue(enq.bits.cloneType, entries, pipe))
    q.io.enq.valid := enq.valid // not using <> so that override is allowed
    q.io.enq.bits := enq.bits
    enq.ready := q.io.enq.ready
    q.io.deq
  }
}
