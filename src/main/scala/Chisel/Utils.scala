// See LICENSE for license details.

package Chisel
import Builder._
import scala.math._
import scala.language.reflectiveCalls
import scala.language.experimental.macros
import scala.reflect.runtime.universe._
import scala.reflect.macros.blackbox._

object Enum {
  /** Returns a sequence of Bits subtypes with values from 0 until n. Helper method. */
  private def createValues[T <: Bits](nodeType: T, n: Int): Seq[T] = (0 until n).map(x => nodeType.fromInt(x))

  /** create n enum values of given type */
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = createValues(nodeType, n).toList

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: Symbol *): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap

  /** create enum values of given type and names */
  def apply[T <: Bits](nodeType: T, l: List[Symbol]): Map[Symbol, T] = (l zip createValues(nodeType, l.length)).toMap
}

/** Compute the log2 rounded up with min value of 1 */
object log2Up {
  def apply(in: Int): Int = 1 max BigInt(in-1).bitLength
}

/** Compute the log2 rounded up */
object log2Ceil {
  def apply(in: Int): Int = {
    require(in > 0)
    BigInt(in-1).bitLength
  }
}

/** Compute the log2 rounded down with min value of 1 */
object log2Down {
  def apply(in: Int): Int = log2Up(in) - (if (isPow2(in)) 0 else 1)
}

/** Compute the log2 rounded down */
object log2Floor {
  def apply(in: Int): Int = log2Ceil(in) - (if (isPow2(in)) 0 else 1)
}

/** Check if an Integer is a power of 2 */
object isPow2 {
  def apply(in: Int): Boolean = in > 0 && ((in & (in-1)) == 0)
}

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

object RegNext {

  def apply[T <: Data](next: T): T = Reg[T](next, next, null.asInstanceOf[T])

  def apply[T <: Data](next: T, init: T): T = Reg[T](next, next, init)

}

object RegInit {

  def apply[T <: Data](init: T): T = Reg[T](init, null.asInstanceOf[T], init)

}

/** A register with an Enable signal */
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
  def apply[T <: Data](sel: Seq[Bool], in: Seq[T]): T =
    apply(sel zip in)
  def apply[T <: Data](in: Iterable[(Bool, T)]): T = {
    if (in.tail.isEmpty) {
      in.head._2
    } else {
      val masked = in map {case (s, i) => Mux(s, i.toBits, Bits(0))}
      val width = in.map(_._2.width).reduce(_ max _)
      in.head._2.cloneTypeWidth(width).fromBits(masked.reduceLeft(_|_))
    }
  }
  def apply[T <: Data](sel: UInt, in: Seq[T]): T =
    apply((0 until in.size).map(sel(_)), in)
  def apply(sel: UInt, in: UInt): Bool = (sel & in).orR
}

/** Builds a Mux tree under the assumption that multiple select signals
  can be enabled. Priority is given to the first select signal.

  Returns the output of the Mux tree.
  */
object PriorityMux
{
  def apply[T <: Bits](in: Seq[(Bool, T)]): T = {
    if (in.size == 1) {
      in.head._2
    } else {
      Mux(in.head._1, in.head._2, apply(in.tail))
    }
  }
  def apply[T <: Bits](sel: Seq[Bool], in: Seq[T]): T = apply(sel zip in)
  def apply[T <: Bits](sel: Bits, in: Seq[T]): T = apply((0 until in.size).map(sel(_)), in)
}

/** This is identical to [[Chisel.when when]] with the condition inverted */
object unless {
  def apply(c: Bool)(block: => Unit) {
    when (!c) { block }
  }
}

class SwitchContext[T <: Bits](cond: T) {
  def is(v: Iterable[T])(block: => Unit) {
    if (!v.isEmpty) when (v.map(_.asUInt === cond.asUInt).reduce(_||_)) { block }
  }
  def is(v: T)(block: => Unit) { is(Seq(v))(block) }
  def is(v: T, vr: T*)(block: => Unit) { is(v :: vr.toList)(block) }
}

/** An object for separate cases in [[Chisel.switch switch]]
  * It is equivalent to a [[Chisel.when$ when]] block comparing to the condition
  * Use outside of a switch statement is illegal */
object is { // Begin deprecation of non-type-parameterized is statements.
  def apply(v: Iterable[Bits])(block: => Unit) {
    Builder.error("The 'is' keyword may not be used outside of a switch.")
  }

  def apply(v: Bits)(block: => Unit) {
    Builder.error("The 'is' keyword may not be used outside of a switch.")
  }

  def apply(v: Bits, vr: Bits*)(block: => Unit) {
    Builder.error("The 'is' keyword may not be used outside of a switch.")
  }
}

/** Conditional logic to form a switch block
  * @example
  * {{{ ... // default values here
  * switch ( myState ) {
  *   is( state1 ) {
  *     ... // some logic here
  *   }
  *   is( state2 ) {
  *     ... // some logic here
  *   }
  * } }}}*/
object switch {
  def apply[T <: Bits](cond: T)(x: => Unit): Unit = macro impl
  def impl(c: Context)(cond: c.Tree)(x: c.Tree) = { import c.universe._
    val sc = c.universe.internal.reificationSupport.freshTermName("sc")
    def extractIsStatement(tree: Tree): List[c.universe.Tree] = tree match {
      case q"Chisel.is.apply( ..$params )( ..$body )" => List(q"$sc.is( ..$params )( ..$body )")
      case b => throw new Exception(s"Cannot include blocks that do not begin with is() in switch.")
    }
    val q"..$body" = x
    val ises = body.flatMap(extractIsStatement(_))
    q"""{ val $sc = new SwitchContext($cond); ..$ises }"""
  }
}

/** MuxLookup creates a cascade of n Muxs to search for a key value */
object MuxLookup {
  /** @param key a key to search for
    * @param default a default value if nothing is found
    * @param mapping a sequence to search of keys and values
    * @return the value found or the default if not
    */
  def apply[S <: UInt, T <: Bits] (key: S, default: T, mapping: Seq[(S, T)]): T = {
    var res = default;
    for ((k, v) <- mapping.reverse)
      res = Mux(k === key, v, res);
    res
  }

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

/** MuxCase returns the first value that is enabled in a map of values */
object MuxCase {
  /** @param default the default value if none are enabled
    * @param mapping a set of data values with associated enables
    * @return the first value in mapping that is enabled */
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

/** Returns the n-cycle delayed version of the input signal.
  */
object ShiftRegister
{
  /** @param in input to delay
    * @param n number of cycles to delay
    * @param en enable the shift */
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

/** Compute Log2 with truncation of a UInt in hardware using a Mux Tree
  * An alternative interpretation is it computes the minimum number of bits needed to represent x
  * @example
  * {{{ data_out := Log2(data_in) }}}
  * @note Truncation is used so Log2(UInt(12412)) = 13*/
object Log2 {
  /** Compute the Log2 on the least significant n bits of x */
  def apply(x: Bits, width: Int): UInt = {
    if (width < 2) {
      UInt(0)
    } else if (width == 2) {
      x(1)
    } else {
      Mux(x(width-1), UInt(width-1), apply(x, width-1))
    }
  }

  def apply(x: Bits): UInt = apply(x, x.getWidth)
}

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
      UInt(1) << in
    } else {
      (UInt(1) << in(log2Up(width)-1,0))(width-1,0)
    }
}

/** A counter module
  * @param n The maximum value of the counter, does not have to be power of 2
  */
class Counter(val n: Int) {
  val value = if (n == 1) UInt(0) else Reg(init=UInt(0, log2Up(n)))
  def inc(): Bool = {
    if (n == 1) {
      Bool(true)
    } else {
      val wrap = value === UInt(n-1)
      value := Mux(Bool(!isPow2(n)) && wrap, UInt(0), value + UInt(1))
      wrap
    }
  }
}

/** Counter Object
  * Example Usage:
  * {{{ val countOn = Bool(true) // increment counter every clock cycle
  * val myCounter = Counter(countOn, n)
  * when ( myCounter.value === UInt(3) ) { ... } }}}*/
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

/** An I/O Bundle containing data and a signal determining if it is valid */
class ValidIO[+T <: Data](gen2: T) extends Bundle
{
  val valid = Bool(OUTPUT)
  val bits = gen2.cloneType.asOutput
  def fire(dummy: Int = 0): Bool = valid
  override def cloneType: this.type = new ValidIO(gen2).asInstanceOf[this.type]
}

/** Adds a valid protocol to any interface. The standard used is
  that the consumer uses the flipped interface.
*/
object Valid {
  def apply[T <: Data](gen: T): ValidIO[T] = new ValidIO(gen)
}

/** An I/O Bundle with simple handshaking using valid and ready signals for data 'bits'*/
class DecoupledIO[+T <: Data](gen: T) extends Bundle
{
  val ready = Bool(INPUT)
  val valid = Bool(OUTPUT)
  val bits  = gen.cloneType.asOutput
  def fire(dummy: Int = 0): Bool = ready && valid
  override def cloneType: this.type = new DecoupledIO(gen).asInstanceOf[this.type]
}

/** Adds a ready-valid handshaking protocol to any interface.
  * The standard used is that the consumer uses the flipped interface.
  */
object Decoupled {
  def apply[T <: Data](gen: T): DecoupledIO[T] = new DecoupledIO(gen)
}

/** An I/O bundle for enqueuing data with valid/ready handshaking */
class EnqIO[T <: Data](gen: T) extends DecoupledIO(gen)
{
  def enq(dat: T): T = { valid := Bool(true); bits := dat; dat }
  valid := Bool(false)
  for (io <- bits.flatten)
    io := UInt(0)
  override def cloneType: this.type = { new EnqIO(gen).asInstanceOf[this.type]; }
}

/** An I/O bundle for dequeuing data with valid/ready handshaking */
class DeqIO[T <: Data](gen: T) extends DecoupledIO(gen)
{
  flip()
  ready := Bool(false)
  def deq(b: Boolean = false): T = { ready := Bool(true); bits }
  override def cloneType: this.type = { new DeqIO(gen).asInstanceOf[this.type]; }
}

/** An I/O bundle for dequeuing data with valid/ready handshaking */
class DecoupledIOC[+T <: Data](gen: T) extends Bundle
{
  val ready = Bool(INPUT)
  val valid = Bool(OUTPUT)
  val bits  = gen.cloneType.asOutput
}

/** An I/O Bundle for Queues
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue */
class QueueIO[T <: Data](gen: T, entries: Int) extends Bundle
{
  /** I/O to enqueue data, is [[Chisel.DecoupledIO]] flipped */
  val enq   = Decoupled(gen.cloneType).flip
  /** I/O to enqueue data, is [[Chisel.DecoupledIO]]*/
  val deq   = Decoupled(gen.cloneType)
  /** The current amount of data in the queue */
  val count = UInt(OUTPUT, log2Up(entries + 1))
}

/** A hardware module implementing a Queue
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue
  * @param pipe True if a single entry queue can run at full throughput (like a pipeline). The ''ready'' signals are
  * combinationally coupled.
  * @param flow True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
  * The ''valid'' signals are coupled.
  *
  * Example usage:
  *    {{{ val q = new Queue(UInt(), 16)
  *    q.io.enq <> producer.io.out
  *    consumer.io.in <> q.io.deq }}}
  */
class Queue[T <: Data](gen: T, val entries: Int,
                       pipe: Boolean = false,
                       flow: Boolean = false,
                       _reset: Bool = null) extends Module(_reset=_reset)
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
    io.count := Mux(ptr_match,
                    Mux(maybe_full,
                      UInt(entries), UInt(0)),
                    Mux(deq_ptr.value > enq_ptr.value,
                      UInt(entries) + ptr_diff, ptr_diff))
  }
}

/** Generic hardware queue. Required parameter entries controls
  the depth of the queues. The width of the queue is determined
  from the inputs.

  Example usage:
     {{{ val q = Queue(Decoupled(UInt()), 16)
     q.io.enq <> producer.io.out
     consumer.io.in <> q.io.deq }}}
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

/** Returns a bit vector in which only the least-significant 1 bit in
  the input vector, if any, is set.
  */
object PriorityEncoderOH
{
  private def encode(in: Seq[Bool]): UInt = {
    val outs = Vec.tabulate(in.size)(i => UInt(BigInt(1) << i, in.size))
    PriorityMux(in :+ Bool(true), outs :+ UInt(0, in.size))
  }
  def apply(in: Seq[Bool]): Vec[Bool] = {
    val enc = encode(in)
    Vec.tabulate(in.size)(enc(_))
  }
  def apply(in: Bits): UInt = encode((0 until in.getWidth).map(i => in(i)))
}

/** An I/O bundle for the Arbiter */
class ArbiterIO[T <: Data](gen: T, n: Int) extends Bundle {
  val in  = Vec(Decoupled(gen), n).flip
  val out = Decoupled(gen)
  val chosen = UInt(OUTPUT, log2Up(n))
}

/** Arbiter Control determining which producer has access */
object ArbiterCtrl
{
  def apply(request: Seq[Bool]): Seq[Bool] = {
    Bool(true) +: (1 until request.length).map(i => !request.slice(0, i).foldLeft(Bool(false))(_ || _))
  }
}

abstract class LockingArbiterLike[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends Module {
  require(isPow2(count))
  def grant: Seq[Bool]
  val io = new ArbiterIO(gen, n)
  val locked  = if(count > 1) Reg(init=Bool(false)) else Bool(false)
  val lockIdx = if(count > 1) Reg(init=UInt(n-1)) else UInt(n-1)
  val chosen = Wire(UInt(width = log2Up(n)))

  io.out.valid := io.in(chosen).valid
  io.out.bits := io.in(chosen).bits
  io.chosen := chosen

  io.in(chosen).ready := Bool(false) // XXX FIRRTL workaround
  for ((g, i) <- grant.zipWithIndex)
    io.in(i).ready := Mux(locked, lockIdx === UInt(i), g) && io.out.ready

  if(count > 1){
    val cnt = Reg(init=UInt(0, width = log2Up(count)))
    val cnt_next = cnt + UInt(1)
    when(io.out.fire()) {
      when(needsLock.map(_(io.out.bits)).getOrElse(Bool(true))) {
        cnt := cnt_next
        when(!locked) {
          locked := Bool(true)
          lockIdx := Vec(io.in.map{ in => in.fire()}).indexWhere{i: Bool => i}
        }
      }
      when(cnt_next === UInt(0)) {
        locked := Bool(false)
      }
    }
  }
}

class LockingRRArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  lazy val last_grant = Reg(init=UInt(0, log2Up(n)))
  override def grant: Seq[Bool] = {
    val ctrl = ArbiterCtrl((0 until n).map(i => io.in(i).valid && UInt(i) > last_grant) ++ io.in.map(_.valid))
    (0 until n).map(i => ctrl(i) && UInt(i) > last_grant || ctrl(i + n))
  }

  var choose = UInt(n-1)
  for (i <- n-2 to 0 by -1)
    choose = Mux(io.in(i).valid, UInt(i), choose)
  for (i <- n-1 to 1 by -1)
    choose = Mux(io.in(i).valid && UInt(i) > last_grant, UInt(i), choose)
  chosen := Mux(locked, lockIdx, choose)

  when (io.out.fire()) { last_grant := chosen }
}

class LockingArbiter[T <: Data](gen: T, n: Int, count: Int, needsLock: Option[T => Bool] = None)
    extends LockingArbiterLike[T](gen, n, count, needsLock) {
  def grant: Seq[Bool] = ArbiterCtrl(io.in.map(_.valid))

  var choose = UInt(n-1)
  for (i <- n-2 to 0 by -1) {
    choose = Mux(io.in(i).valid, UInt(i), choose)
  }
  chosen := Mux(locked, lockIdx, choose)
}

/** Hardware module that is used to sequence n producers into 1 consumer.
  Producers are chosen in round robin order.

  Example usage:
    val arb = new RRArbiter(2, UInt())
    arb.io.in(0) <> producer0.io.out
    arb.io.in(1) <> producer1.io.out
    consumer.io.in <> arb.io.out
  */
class RRArbiter[T <: Data](gen:T, n: Int) extends LockingRRArbiter[T](gen, n, 1)

/** Hardware module that is used to sequence n producers into 1 consumer.
 Priority is given to lower producer

 Example usage:
   val arb = Module(new Arbiter(2, UInt()))
   arb.io.in(0) <> producer0.io.out
   arb.io.in(1) <> producer1.io.out
   consumer.io.in <> arb.io.out
 */
class Arbiter[T <: Data](gen: T, n: Int) extends LockingArbiter[T](gen, n, 1)

/** linear feedback shift register
  */
object LFSR16
{
  def apply(increment: Bool = Bool(true)): UInt =
  {
    val width = 16
    val lfsr = Reg(init=UInt(1, width))
    when (increment) { lfsr := Cat(lfsr(0)^lfsr(2)^lfsr(3)^lfsr(5), lfsr(width-1,1)) }
    lfsr
  }
}

/** A hardware module that delays data coming down the pipeline
  by the number of cycles set by the latency parameter. Functionality
  is similar to ShiftRegister but this exposes a Pipe interface.

  Example usage:
    val pipe = new Pipe(UInt())
    pipe.io.enq <> produce.io.out
    consumer.io.in <> pipe.io.deq
  */
object Pipe
{
  def apply[T <: Data](enqValid: Bool, enqBits: T, latency: Int): ValidIO[T] = {
    if (latency == 0) {
      val out = Wire(Valid(enqBits))
      out.valid <> enqValid
      out.bits <> enqBits
      out
    } else {
      val v = Reg(Bool(), next=enqValid, init=Bool(false))
      val b = RegEnable(enqBits, enqValid)
      apply(v, b, latency-1)
    }
  }
  def apply[T <: Data](enqValid: Bool, enqBits: T): ValidIO[T] = apply(enqValid, enqBits, 1)
  def apply[T <: Data](enq: ValidIO[T], latency: Int = 1): ValidIO[T] = apply(enq.valid, enq.bits, latency)
}

class Pipe[T <: Data](gen: T, latency: Int = 1) extends Module
{
  val io = new Bundle {
    val enq = Valid(gen).flip
    val deq = Valid(gen)
  }

  io.deq <> Pipe(io.enq, latency)
}
