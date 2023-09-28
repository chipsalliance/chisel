// SPDX-License-Identifier: Apache-2.0

/** Wrappers for ready-valid (Decoupled) interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.{requireIsChiselType, Direction}
import chisel3.reflect.DataMirror
import chisel3.util.simpleClassName

import scala.annotation.nowarn

/** An I/O Bundle containing 'valid' and 'ready' signals that handshake
  * the transfer of data stored in the 'bits' subfield.
  * The base protocol implied by the directionality is that
  * the producer uses the interface as-is (outputs bits)
  * while the consumer uses the flipped interface (inputs bits).
  * The actual semantics of ready/valid are enforced via the use of concrete subclasses.
  * @param gen the type of data to be wrapped in Ready/Valid
  * @groupdesc Signals The actual hardware fields of the Bundle
  */
abstract class ReadyValidIO[+T <: Data](gen: T) extends Bundle {

  /** Indicates that the consumer is ready to accept the data this cycle
    * @group Signals
    */
  val ready = Input(Bool())

  /** Indicates that the producer has put valid data in 'bits'
    * @group Signals
    */
  val valid = Output(Bool())

  /** The data to be transferred when ready and valid are asserted at the same cycle
    * @group Signals
    */
  val bits = Output(gen)

  /** A stable typeName for this `ReadyValidIO` and any of its implementations
    * using the supplied `Data` generator's `typeName`
    */
  override def typeName = s"${simpleClassName(this.getClass)}_${gen.typeName}"
}

object ReadyValidIO {

  implicit class AddMethodsToReadyValid[T <: Data](target: ReadyValidIO[T]) {

    /** Indicates if IO is both ready and valid
      */
    def fire: Bool = target.ready && target.valid

    /** Push dat onto the output bits of this interface to let the consumer know it has happened.
      * @param dat the values to assign to bits.
      * @return    dat.
      */
    def enq(dat: T): T = {
      target.valid := true.B
      target.bits := dat
      dat
    }

    /** Indicate no enqueue occurs. Valid is set to false, and bits are
      * connected to an uninitialized wire.
      */
    def noenq(): Unit = {
      target.valid := false.B
      target.bits := DontCare
    }

    /** Assert ready on this port and return the associated data bits.
      * This is typically used when valid has been asserted by the producer side.
      * @return The data bits.
      */
    def deq(): T = {
      target.ready := true.B
      target.bits
    }

    /** Indicate no dequeue occurs. Ready is set to false.
      */
    def nodeq(): Unit = {
      target.ready := false.B
    }
  }
}

/** A concrete subclass of ReadyValidIO signaling that the user expects a
  * "decoupled" interface: 'valid' indicates that the producer has
  * put valid data in 'bits', and 'ready' indicates that the consumer is ready
  * to accept the data this cycle. No requirements are placed on the signaling
  * of ready or valid.
  * @param gen the type of data to be wrapped in DecoupledIO
  */
class DecoupledIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen) {

  /** Applies the supplied functor to the bits of this interface, returning a new
    * typed DecoupledIO interface.
    * @param f The function to apply to this DecoupledIO's 'bits' with return type B
    * @return a new DecoupledIO of type B
    */
  def map[B <: Data](f: T => B): DecoupledIO[B] = {
    val _map_bits = f(bits)
    val _map = Wire(Decoupled(chiselTypeOf(_map_bits)))
    _map.bits := _map_bits
    _map.valid := valid
    ready := _map.ready
    _map
  }
}

/** This factory adds a decoupled handshaking protocol to a data bundle. */
object Decoupled {

  /** Wraps some Data with a DecoupledIO interface. */
  def apply[T <: Data](gen: T): DecoupledIO[T] = new DecoupledIO(gen)

  // TODO: use a proper empty data type, this is a quick and dirty solution
  private final class EmptyBundle extends Bundle

  // Both of these methods return DecoupledIO parameterized by the most generic type: Data
  /** Returns a [[DecoupledIO]] inteface with no payload */
  def apply(): DecoupledIO[Data] = apply(new EmptyBundle)

  /** Returns a [[DecoupledIO]] inteface with no payload */
  def empty: DecoupledIO[Data] = Decoupled()

  /** Downconverts an IrrevocableIO output to a DecoupledIO, dropping guarantees of irrevocability.
    *
    * @note unsafe (and will error) on the producer (input) side of an IrrevocableIO
    */
  def apply[T <: Data](irr: IrrevocableIO[T]): DecoupledIO[T] = {
    require(
      DataMirror.directionOf(irr.bits) == Direction.Output,
      "Only safe to cast produced Irrevocable bits to Decoupled."
    )
    val d = Wire(new DecoupledIO(chiselTypeOf(irr.bits)))
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
  * @param gen the type of data to be wrapped in IrrevocableIO
  * @groupdesc Signals The actual hardware fields of the Bundle
  */
class IrrevocableIO[+T <: Data](gen: T) extends ReadyValidIO[T](gen)

/** Factory adds an irrevocable handshaking protocol to a data bundle. */
object Irrevocable {
  def apply[T <: Data](gen: T): IrrevocableIO[T] = new IrrevocableIO(gen)

  /** Upconverts a DecoupledIO input to an IrrevocableIO, allowing an IrrevocableIO to be used
    * where a DecoupledIO is expected.
    *
    * @note unsafe (and will error) on the consumer (output) side of an DecoupledIO
    */
  def apply[T <: Data](dec: DecoupledIO[T]): IrrevocableIO[T] = {
    require(
      DataMirror.directionOf(dec.bits) == Direction.Input,
      "Only safe to cast consumed Decoupled bits to Irrevocable."
    )
    val i = Wire(new IrrevocableIO(chiselTypeOf(dec.bits)))
    dec.bits := i.bits
    dec.valid := i.valid
    i.ready := dec.ready
    i
  }
}

/** Producer - drives (outputs) valid and bits, inputs ready.
  * @param gen The type of data to enqueue
  */
object EnqIO {
  def apply[T <: Data](gen: T): DecoupledIO[T] = Decoupled(gen)
}

/** Consumer - drives (outputs) ready, inputs valid and bits.
  * @param gen The type of data to dequeue
  */
object DeqIO {
  def apply[T <: Data](gen: T): DecoupledIO[T] = Flipped(Decoupled(gen))
}

/** An I/O Bundle for Queues
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue.
  * @param hasFlush A boolean for whether the generated Queue is flushable
  * @groupdesc Signals The hardware fields of the Bundle
  */
class QueueIO[T <: Data](
  private val gen: T,
  val entries:     Int,
  val hasFlush:    Boolean = false)
    extends Bundle { // See github.com/freechipsproject/chisel3/issues/765 for why gen is a private val and proposed replacement APIs.

  /* These may look inverted, because the names (enq/deq) are from the perspective of the client,
   *  but internally, the queue implementation itself sits on the other side
   *  of the interface so uses the flipped instance.
   */
  /** I/O to enqueue data (client is producer, and Queue object is consumer), is [[chisel3.util.DecoupledIO]] flipped.
    * @group Signals
    */
  val enq = Flipped(EnqIO(gen))

  /** I/O to dequeue data (client is consumer and Queue object is producer), is [[chisel3.util.DecoupledIO]]
    * @group Signals
    */
  val deq = Flipped(DeqIO(gen))

  /** The current amount of data in the queue
    * @group Signals
    */
  val count = Output(UInt(log2Ceil(entries + 1).W))

  /** When asserted, reset the enqueue and dequeue pointers, effectively flushing the queue (Optional IO for a flushable Queue)
    * @group Signals
    */
  val flush = if (hasFlush) Some(Input(Bool())) else None

}

/** A hardware module implementing a Queue
  * @param gen The type of data to queue
  * @param entries The max number of entries in the queue
  * @param pipe True if a single entry queue can run at full throughput (like a pipeline). The ''ready'' signals are
  * combinationally coupled.
  * @param flow True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
  * The ''valid'' signals are coupled.
  * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element.
  * @param hasFlush True if generated queue requires a flush feature
  * @example {{{
  * val q = Module(new Queue(UInt(), 16))
  * q.io.enq <> producer.io.out
  * consumer.io.in <> q.io.deq
  * }}}
  */
class Queue[T <: Data](
  val gen:            T,
  val entries:        Int,
  val pipe:           Boolean = false,
  val flow:           Boolean = false,
  val useSyncReadMem: Boolean = false,
  val hasFlush:       Boolean = false)
    extends Module() {
  require(entries > -1, "Queue must have non-negative number of entries")
  require(entries != 0, "Use companion object Queue.apply for zero entries")
  requireIsChiselType(gen)

  val io = IO(new QueueIO(gen, entries, hasFlush))
  val ram = if (useSyncReadMem) SyncReadMem(entries, gen, SyncReadMem.WriteFirst) else Mem(entries, gen)
  val enq_ptr = Counter(entries)
  val deq_ptr = Counter(entries)
  val maybe_full = RegInit(false.B)
  val ptr_match = enq_ptr.value === deq_ptr.value
  val empty = ptr_match && !maybe_full
  val full = ptr_match && maybe_full
  val do_enq = WireDefault(io.enq.fire)
  val do_deq = WireDefault(io.deq.fire)
  val flush = io.flush.getOrElse(false.B)

  // when flush is high, empty the queue
  // Semantically, any enqueues happen before the flush.
  when(do_enq) {
    ram(enq_ptr.value) := io.enq.bits
    enq_ptr.inc()
  }
  when(do_deq) {
    deq_ptr.inc()
  }
  when(do_enq =/= do_deq) {
    maybe_full := do_enq
  }
  when(flush) {
    enq_ptr.reset()
    deq_ptr.reset()
    maybe_full := false.B
  }

  io.deq.valid := !empty
  io.enq.ready := !full

  if (useSyncReadMem) {
    val deq_ptr_next = Mux(deq_ptr.value === (entries.U - 1.U), 0.U, deq_ptr.value + 1.U)
    val r_addr = WireDefault(Mux(do_deq, deq_ptr_next, deq_ptr.value))
    io.deq.bits := ram.read(r_addr)
  } else {
    io.deq.bits := ram(deq_ptr.value)
  }

  if (flow) {
    when(io.enq.valid) { io.deq.valid := true.B }
    when(empty) {
      io.deq.bits := io.enq.bits
      do_deq := false.B
      when(io.deq.ready) { do_enq := false.B }
    }
  }

  if (pipe) {
    when(io.deq.ready) { io.enq.ready := true.B }
  }

  val ptr_diff = enq_ptr.value - deq_ptr.value

  if (isPow2(entries)) {
    io.count := Mux(maybe_full && ptr_match, entries.U, 0.U) | ptr_diff
  } else {
    io.count := Mux(
      ptr_match,
      Mux(maybe_full, entries.asUInt, 0.U),
      Mux(deq_ptr.value > enq_ptr.value, entries.asUInt + ptr_diff, ptr_diff)
    )
  }

  /** Give this Queue a default, stable desired name using the supplied `Data`
    * generator's `typeName`
    */
  override def desiredName = s"Queue${entries}_${gen.typeName}"
}

/** Factory for a generic hardware queue. */
object Queue {

  /** Create a [[Queue]] and supply a [[DecoupledIO]] containing the product.
    *
    * @param enq input (enqueue) interface to the queue, also determines type of queue elements.
    * @param entries depth (number of elements) of the queue
    * @param pipe True if a single entry queue can run at full throughput (like a pipeline). The `ready` signals are
    *             combinationally coupled.
    * @param flow True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
    *             The `valid` signals are coupled.
    * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element.
    * @param flush Optional [[Bool]] signal, if defined, the [[Queue.hasFlush]] will be true, and connect correspond
    *              signal to [[Queue]] instance.
    * @return output (dequeue) interface from the queue.
    *
    * @example {{{
    *   consumer.io.in <> Queue(producer.io.out, 16)
    * }}}
    */
  @nowarn("cat=deprecation&msg=TransitName")
  def apply[T <: Data](
    enq:            ReadyValidIO[T],
    entries:        Int = 2,
    pipe:           Boolean = false,
    flow:           Boolean = false,
    useSyncReadMem: Boolean = false,
    flush:          Option[Bool] = None
  ): DecoupledIO[T] = {
    if (entries == 0) {
      val deq = Wire(new DecoupledIO(chiselTypeOf(enq.bits)))
      deq.valid := enq.valid
      deq.bits := enq.bits
      enq.ready := deq.ready
      deq
    } else {
      val q = Module(new Queue(chiselTypeOf(enq.bits), entries, pipe, flow, useSyncReadMem, flush.isDefined))
      q.io.flush.zip(flush).foreach(f => f._1 := f._2)
      q.io.enq.valid := enq.valid // not using <> so that override is allowed
      q.io.enq.bits := enq.bits
      enq.ready := q.io.enq.ready
      q.io.deq
    }
  }

  /** Create a queue and supply a [[IrrevocableIO]] containing the product.
    * Casting from [[DecoupledIO]] is safe here because we know the [[Queue]] has
    * Irrevocable semantics.
    * we didn't want to change the return type of apply() for backwards compatibility reasons.
    *
    * @param enq [[DecoupledIO]] signal to enqueue.
    * @param entries The max number of entries in the queue
    * @param pipe True if a single entry queue can run at full throughput (like a pipeline). The ''ready'' signals are
    * combinationally coupled.
    * @param flow True if the inputs can be consumed on the same cycle (the inputs "flow" through the queue immediately).
    * The ''valid'' signals are coupled.
    * @param useSyncReadMem True uses SyncReadMem instead of Mem as an internal memory element.
    * @param flush Optional [[Bool]] signal, if defined, the [[Queue.hasFlush]] will be true, and connect correspond
    *              signal to [[Queue]] instance.
    * @return a [[DecoupledIO]] signal which should connect to the dequeue signal.
    *
    * @example {{{
    *   consumer.io.in <> Queue(producer.io.out, 16)
    * }}}
    */
  def irrevocable[T <: Data](
    enq:            ReadyValidIO[T],
    entries:        Int = 2,
    pipe:           Boolean = false,
    flow:           Boolean = false,
    useSyncReadMem: Boolean = false,
    flush:          Option[Bool] = None
  ): IrrevocableIO[T] = {
    val deq = apply(enq, entries, pipe, flow, useSyncReadMem, flush)
    require(entries > 0, "Zero-entry queues don't guarantee Irrevocability")
    val irr = Wire(new IrrevocableIO(chiselTypeOf(deq.bits)))
    irr.bits := deq.bits
    irr.valid := deq.valid
    deq.ready := irr.ready
    irr
  }
}
