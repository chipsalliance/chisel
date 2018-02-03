// See LICENSE for license details.

/** Wrappers for valid interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.core.CompileOptions
import chisel3.experimental.DataMirror
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order

/** An Bundle containing data and a signal determining if it is valid */
class Valid[+T <: Data](gen: T) extends Bundle
{
  val valid = Output(Bool())
  val bits  = Output(gen)
  def fire(dummy: Int = 0): Bool = valid
  override def cloneType: this.type = Valid(gen).asInstanceOf[this.type]
}

/** Adds a valid protocol to any interface */
object Valid {
  def apply[T <: Data](gen: T): Valid[T] = new Valid(gen)
}

/** A hardware module that delays data coming down the pipeline
  * by the number of cycles set by the latency parameter. Functionality
  * is similar to ShiftRegister but this exposes a Pipe interface.
  *
  * Example usage:
  * {{{
  *   val pipe = new Pipe(UInt())
  *   pipe.io.enq <> produce.io.out
  *   consumer.io.in <> pipe.io.deq
  * }}}
  */
object Pipe
{
  @chiselName
  def apply[T <: Data](enqValid: Bool, enqBits: T, latency: Int)(implicit compileOptions: CompileOptions): Valid[T] = {
    if (latency == 0) {
      val out = Wire(Valid(chiselTypeOf(enqBits)))
      out.valid := enqValid
      out.bits := enqBits
      out
    } else {
      val v = RegNext(enqValid, false.B)
      val b = RegEnable(enqBits, enqValid)
      apply(v, b, latency-1)(compileOptions)
    }
  }
  def apply[T <: Data](enqValid: Bool, enqBits: T)(implicit compileOptions: CompileOptions): Valid[T] = {
    apply(enqValid, enqBits, 1)(compileOptions)
  }
  def apply[T <: Data](enq: Valid[T], latency: Int = 1)(implicit compileOptions: CompileOptions): Valid[T] = {
    apply(enq.valid, enq.bits, latency)(compileOptions)
  }
}

class Pipe[T <: Data](gen: T, latency: Int = 1)(implicit compileOptions: CompileOptions) extends Module
{
  class PipeIO extends Bundle {
    val enq = Input(Valid(gen))
    val deq = Output(Valid(gen))
  }

  val io = IO(new PipeIO)

  io.deq <> Pipe(io.enq, latency)
}
