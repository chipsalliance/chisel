// See LICENSE for license details.

/** Wrappers for ready-valid (Decoupled) interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._

/** An I/O Bundle containing 'valid' and 'ready' signals that handshake
  * the transfer of data stored in the 'bits' subfield.
  * The base protocol implied by the directionality is that the consumer
  * uses the flipped interface. Actual semantics of ready/valid are
  * enforced via use of concrete subclasses.
  */
abstract class ReadyValidIO[+T <: Data](gen: T) extends Bundle
{
  val ready = Bool(INPUT)
  val valid = Bool(OUTPUT)
  val bits  = gen.cloneType.asOutput
  def fire(dummy: Int = 0): Bool = ready && valid
}

/** A concrete subclass of ReadyValidIO signalling that the user expects a
  * "decoupled" interface: 'valid' indicates that the producer has
  * put valid data in 'bits', and 'ready' indicates that the consumer is ready
  * to accept the data this cycle. No requirements are placed on the signalling
  * of ready or valid.
  */
class DecoupledIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen)
{
  override def cloneType: this.type = new DecoupledIO(gen).asInstanceOf[this.type]
}

/** This factory adds a decoupled handshaking protocol to a data bundle. */
object Decoupled
{
  /** Take any Data and wrap it in a DecoupledIO interface */
  def apply[T <: Data](gen: T): DecoupledIO[T] = new DecoupledIO(gen)

  /** Take an IrrevocableIO and cast it to a DecoupledIO.
    * This cast is only safe to do in cases where the IrrevocableIO
    * is being produced as an output.
    */
  def apply[T <: Data](irr: IrrevocableIO[T]): DecoupledIO[T] = {
    require(irr.bits.dir == OUTPUT, "Only safe to cast produced Irrevocable bits to Decoupled.")
    val d = Wire(new DecoupledIO(irr.bits))
    d.bits := irr.bits
    d.valid := irr.valid
    irr.ready := d.ready
    d
  }
}

/** A concrete subclass of ReadyValidIO that promises to not change
  * the value of 'bits' after a cycle where 'valid' is high and 'ready' is low.
  * Additionally, once 'valid' is raised it will never be lowered until after
  * 'ready' has also been raised.
  */
class IrrevocableIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen)
{
  override def cloneType: this.type = new IrrevocableIO(gen).asInstanceOf[this.type]
}

/** Factory adds an irrevocable handshaking protocol to a data bundle. */
object Irrevocable
{
  def apply[T <: Data](gen: T): IrrevocableIO[T] = new IrrevocableIO(gen)
}


/** An I/O bundle for enqueuing data with valid/ready handshaking
  * Initialization must be handled, if necessary, by the parent circuit
  */
class EnqIO[T <: Data](gen: T) extends DecoupledIO(gen)
{
  /** push dat onto the output bits of this interface to let the consumer know it has happened.
    * @param dat the values to assign to bits.
    * @return    dat.
    */
  def enq(dat: T): T = { valid := Bool(true); bits := dat; dat }

  /** Initialize this Bundle.  Valid is set to false, and all bits are set to zero.
    * NOTE: This method of initialization is still being discussed and could change in the
    * future.
    */
  def init(): Unit = {
    valid := Bool(false)
    for (io <- bits.flatten)
      io := UInt(0)
  }
  override def cloneType: this.type = { new EnqIO(gen).asInstanceOf[this.type]; }
}

/** An I/O bundle for dequeuing data with valid/ready handshaking.
  * Initialization must be handled, if necessary, by the parent circuit
  */
class DeqIO[T <: Data](gen: T) extends DecoupledIO(gen) with Flipped
{
  /** Assert ready on this port and return the associated data bits.
    * This is typically used when valid has been asserted by the producer side.
    * @param b ignored
    * @return the data for this device,
    */
  def deq(b: Boolean = false): T = { ready := Bool(true); bits }

  /** Initialize this Bundle.
    * NOTE: This method of initialization is still being discussed and could change in the
    * future.
    */
  def init(): Unit = {
    ready := Bool(false)
  }
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
  val enq   = Decoupled(gen.cloneType).flip()
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
class Queue[T <: Data](gen: T,
                       val entries: Int,
                       pipe: Boolean = false,
                       flow: Boolean = false,
                       override_reset: Option[Bool] = None)
extends Module(override_reset=override_reset) {
  def this(gen: T, entries: Int, pipe: Boolean, flow: Boolean, _reset: Bool) =
    this(gen, entries, pipe, flow, Some(_reset))

  val io = new QueueIO(gen, entries)

  val ram = Mem(entries, gen)
  val enq_ptr = Counter(entries)
  val deq_ptr = Counter(entries)
  val maybe_full = Reg(init=Bool(false))

  val ptr_match = enq_ptr.value === deq_ptr.value
  val empty = ptr_match && !maybe_full
  val full = ptr_match && maybe_full
  val do_enq = Wire(init=io.enq.fire())
  val do_deq = Wire(init=io.deq.fire())

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

  io.deq.valid := !empty
  io.enq.ready := !full
  io.deq.bits := ram(deq_ptr.value)

  if (flow) {
    when (io.enq.valid) { io.deq.valid := Bool(true) }
    when (empty) {
      io.deq.bits := io.enq.bits
      do_deq := Bool(false)
      when (io.deq.ready) { do_enq := Bool(false) }
    }
  }

  if (pipe) {
    when (io.deq.ready) { io.enq.ready := Bool(true) }
  }

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

/** Factory for a generic hardware queue. Required parameter 'entries' controls
  * the depth of the queues. The width of the queue is determined
  * from the input 'enq'.
  *
  * Example usage:
  *   {{{ consumer.io.in <> Queue(producer.io.out, 16) }}}
  */
object Queue
{
  /** Create a queue and supply a DecoupledIO containing the product. */
  def apply[T <: Data](
      enq: ReadyValidIO[T],
      entries: Int = 2,
      pipe: Boolean = false,
      flow: Boolean = false): DecoupledIO[T] = {
    val q = Module(new Queue(enq.bits.cloneType, entries, pipe, flow))
    q.io.enq.valid := enq.valid // not using <> so that override is allowed
    q.io.enq.bits := enq.bits
    enq.ready := q.io.enq.ready
    TransitName(q.io.deq, q)
  }

  /** Create a queue and supply a IrrevocableIO containing the product.
    * Casting from Decoupled is safe here because we know the Queue has
    * Irrevocable semantics; we didn't want to change the return type of
    * apply() for backwards compatibility reasons.
    */
  def irrevocable[T <: Data](
      enq: ReadyValidIO[T],
      entries: Int = 2,
      pipe: Boolean = false,
      flow: Boolean = false): IrrevocableIO[T] = {
    val deq = apply(enq, entries, pipe, flow)
    val irr = Wire(new IrrevocableIO(deq.bits))
    irr.bits := deq.bits
    irr.valid := deq.valid
    deq.ready := irr.ready
    irr
  }
}
