// See LICENSE for license details.

/** Wrappers for ready-valid (Decoupled) interfaces and associated circuit generators using them.
  */

package Chisel

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

  val ram = Mem(entries, gen)
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
