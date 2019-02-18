// See LICENSE for license details.

/** Wrappers for valid interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.core.CompileOptions
import chisel3.experimental.DataMirror
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order

/** A [[Bundle]] that adds a `valid` bit to some data. This indicates that the user expects a "valid" interface between
  * a producer and a consumer. Here, the producer asserts the `valid` bit when data on the `bits` line contains valid
  * data. This differs from [[DecoupledIO]] or [[IrrevocableIO]] as there is no `ready` line that the consumer can use
  * to put back pressure on the producer.
  *
  * In most scenarios, the `Valid` class will ''not'' be used directly. Instead, users will create `Valid` interfaces
  * using the [[Valid$ Valid factory]].
  * @tparam T the type of the data
  * @param gen some data
  * @see [[Valid$ Valid factory]] for concrete examples.
  */
class Valid[+T <: Data](gen: T) extends Bundle {
  /** A bit that will be asserted when `bits` is valid */
  val valid = Output(Bool())

  /** Some data */
  val bits  = Output(gen)

  /** True when `valid` is asserted
    * @return a Chisel [[Bool]] true if `valid` is asserted
    */
  def fire(dummy: Int = 0): Bool = valid

  override def cloneType: this.type = Valid(gen).asInstanceOf[this.type]
}

/** Factory for generating "valid" interfaces. A "valid" interface is a data-communicating interface between a producer
  * and a consumer where the producer does not wait for the consumer. Concretely, this means that one additional bit is
  * added to the data indicating its validity.
  *
  * As an example, consider the following [[Bundle]], `MyBundle`:
  * {{{
  *   class MyBundle extends Bundle {
  *     val foo = Output(UInt(8.W))
  *   }
  * }}}
  *
  * To convert this to a "valid" interface, you wrap it with a call to the [[Valid$.apply `Valid` companion object's
  * apply method]]:
  * {{{
  *   val bar = Valid(new MyBundle)
  * }}}
  *
  * The resulting interface is ''structurally'' equivalent to the following:
  * {{{
  *   class MyValidBundle extends Bundle {
  *     val valid = Output(Bool())
  *     val bits = Output(new MyBundle)
  *   }
  * }}}
  *
  * In addition to adding the `valid` bit, a [[Valid.fire]] method is also added that returns the `valid` bit. This
  * provides a similarly named interface to [[DecoupledIO.fire]].
  *
  * @see [[Decoupled$ DecoupledIO Factory]]
  * @see [[Irrevocable$ IrrevocableIO Factory]]
  */
object Valid {

  /** Wrap some [[Data]] in a valid interface
    * @tparam T the type of the data to wrap
    * @param gen the data to wrap
    * @return the wrapped input data
    */
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
object Pipe {
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

class Pipe[T <: Data](gen: T, latency: Int = 1)(implicit compileOptions: CompileOptions) extends Module {
  class PipeIO extends Bundle {
    val enq = Input(Valid(gen))
    val deq = Output(Valid(gen))
  }

  val io = IO(new PipeIO)

  io.deq <> Pipe(io.enq, latency)
}
