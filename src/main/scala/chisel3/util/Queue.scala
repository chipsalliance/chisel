// SPDX-License-Identifier: Apache-2.0
package chisel3.util

import chisel3._
import chisel3.experimental.requireIsChiselType

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
