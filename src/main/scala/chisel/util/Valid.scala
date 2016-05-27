// See LICENSE for license details.

/** Wrappers for valid interfaces and associated circuit generators using them.
  */

package chisel.util

import chisel._

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
