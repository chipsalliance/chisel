// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.prefix
import chisel3.util.simpleClassName

/** A factory to generate a hardware pipe. This can be used to delay [[Valid]] data by a design-time configurable number
  * of cycles.
  *
  * Here, we construct three different pipes using the different provided `apply` methods and hook them up together. The
  * types are explicitly specified to show that these all communicate using [[Valid]] interfaces:
  * {{{
  *   val in: Valid[UInt]  = Wire(Valid(UInt(2.W)))
  *
  *   /* A zero latency (combinational) pipe is connected to 'in' */
  *   val foo: Valid[UInt] = Pipe(in.valid, in.bits, 0)
  *
  *   /* A one-cycle pipe is connected to the output of 'foo' */
  *   val bar: Valid[UInt] = Pipe(foo.valid, foo.bits)
  *
  *   /* A two-cycle pipe is connected to the output of 'bar' */
  *   val baz: Valid[UInt] = Pipe(bar, 2)
  * }}}
  *
  * @see [[Pipe Pipe class]] for an alternative API
  * @see [[Valid]] interface
  * @see [[Queue]] and the [[Queue$ Queue factory]] for actual queues
  * @see The [[ShiftRegister$ ShiftRegister factory]] to generate a pipe without a [[Valid]] interface
  * @define returnType the [[Valid]] output of the final pipeline stage
  */
object Pipe {

  /** Generate a pipe from an explicit valid bit and some data
    * @param enqValid the valid bit (must be a hardware type)
    * @param enqBits the data (must be a hardware type)
    * @param latency the number of pipeline stages
    * @return $returnType
    */
  def apply[T <: Data](enqValid: Bool, enqBits: T, latency: Int): Valid[T] = {
    require(latency >= 0, "Pipe latency must be greater than or equal to zero!")
    if (latency == 0) {
      val out = Wire(Valid(chiselTypeOf(enqBits)))
      out.valid := enqValid
      out.bits := enqBits
      out
    } else
      prefix("pipe") {
        val v = RegNext(enqValid, false.B)
        val b = RegEnable(enqBits, enqValid)
        apply(v, b, latency - 1)
      }
  }

  /** Generate a one-stage pipe from an explicit valid bit and some data
    * @param enqValid the valid bit (must be a hardware type)
    * @param enqBits the data (must be a hardware type)
    * @return $returnType
    */
  def apply[T <: Data](enqValid: Bool, enqBits: T): Valid[T] = {
    apply(enqValid, enqBits, 1)
  }

  /** Generate a pipe for a [[Valid]] interface
    * @param enq a [[Valid]] interface (must be a hardware type)
    * @param latency the number of pipeline stages
    * @return $returnType
    */
  def apply[T <: Data](enq: Valid[T], latency: Int = 1): Valid[T] = {
    apply(enq.valid, enq.bits, latency)
  }
}

/** Pipeline module generator parameterized by data type and latency.
  *
  * This defines a module with one input, `enq`, and one output, `deq`. The input and output are [[Valid]] interfaces
  * that wrap some Chisel type, e.g., a [[UInt]] or a [[Bundle]]. This generator will then chain together a number of
  * pipeline stages that all advance when the input [[Valid]] `enq` fires. The output `deq` [[Valid]] will fire only
  * when valid data has made it all the way through the pipeline.
  *
  * As an example, to construct a 4-stage pipe of 8-bit [[UInt]]s and connect it to a producer and consumer, you can use
  * the following:
  * {{{
  *   val foo = Module(new Pipe(UInt(8.W)), 4)
  *   pipe.io.enq := producer.io
  *   consumer.io := pipe.io.deq
  * }}}
  *
  * If you already have the [[Valid]] input or the components of a [[Valid]] interface, it may be simpler to use the
  * [[Pipe$ Pipe factory]] companion object. This, which [[Pipe]] internally utilizes, will automatically connect the
  * input for you.
  *
  * @param gen a Chisel type
  * @param latency the number of pipeline stages
  * @see [[Pipe$ Pipe factory]] for an alternative API
  * @see [[Valid]] interface
  * @see [[Queue]] and the [[Queue$ Queue factory]] for actual queues
  * @see The [[ShiftRegister$ ShiftRegister factory]] to generate a pipe without a [[Valid]] interface
  */
class Pipe[T <: Data](val gen: T, val latency: Int = 1) extends Module {

  /** A non-ambiguous name of this `Pipe` for use in generated Verilog names.
    * Includes the latency cycle count in the name as well as the parameterized
    * generator's `typeName`, e.g. `Pipe4_UInt4`
    */
  override def desiredName = s"${simpleClassName(this.getClass)}${latency}_${gen.typeName}"

  /** Interface for [[Pipe]]s composed of a [[Valid]] input and [[Valid]] output
    * @define notAQueue
    * @groupdesc Signals Hardware fields of the Bundle
    */
  class PipeIO extends Bundle {

    /** [[Valid]] input
      * @group Signals
      */
    val enq = Input(Valid(gen))

    /** [[Valid]] output. Data will appear here `latency` cycles after being valid at `enq`.
      * @group Signals
      */
    val deq = Output(Valid(gen))
  }

  val io = IO(new PipeIO)

  io.deq <> Pipe(io.enq, latency)
}
