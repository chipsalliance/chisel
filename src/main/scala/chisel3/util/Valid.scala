// See LICENSE for license details.

/** Wrappers for valid interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.core.CompileOptions
import chisel3.experimental.DataMirror

/** A [[Bundle]] that adds a `valid` bit to some data. This indicates that the user expects a "valid" interface between
  * a producer and a consumer. Here, the producer asserts the `valid` bit when data on the `bits` line contains valid
  * data. This differs from [[DecoupledIO]] or [[IrrevocableIO]] as there is no `ready` line that the consumer can use
  * to put back pressure on the producer.
  *
  * In most scenarios, the `Valid` class will ''not'' be used directly. Instead, users will create `Valid` interfaces
  * using the [[Valid$ Valid factory]].
  * @tparam T the type of the data
  * @param gen some data
  * @see [[Valid$ Valid factory]] for concrete examples
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
  * provides a similarly named interface to [[DecoupledIO]]'s fire.
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
  def apply[T <: Data](enqValid: Bool, enqBits: T, latency: Int)(implicit compileOptions: CompileOptions): Valid[T] = {
    require(latency >= 0, "Pipe latency must be greater than or equal to zero!")
    if (latency == 0) {
      val out = Wire(Valid(chiselTypeOf(enqBits)))
      out.valid := enqValid
      out.bits := enqBits
      out
    } else {
      val v = RegNext(enqValid, false.B)
      val b = RegEnable(enqBits, enqValid)
      val out = apply(v, b, latency-1)(compileOptions)

      TransitName.withSuffix("Pipe_valid")(out, v)
      TransitName.withSuffix("Pipe_bits")(out, b)
    }
  }

  /** Generate a one-stage pipe from an explicit valid bit and some data
    * @param enqValid the valid bit (must be a hardware type)
    * @param enqBits the data (must be a hardware type)
    * @return $returnType
    */
  def apply[T <: Data](enqValid: Bool, enqBits: T)(implicit compileOptions: CompileOptions): Valid[T] = {
    apply(enqValid, enqBits, 1)(compileOptions)
  }

  /** Generate a pipe for a [[Valid]] interface
    * @param enq a [[Valid]] interface (must be a hardware type)
    * @param latency the number of pipeline stages
    * @return $returnType
    */
  def apply[T <: Data](enq: Valid[T], latency: Int = 1)(implicit compileOptions: CompileOptions): Valid[T] = {
    apply(enq.valid, enq.bits, latency)(compileOptions)
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
class Pipe[T <: Data](gen: T, latency: Int = 1)(implicit compileOptions: CompileOptions) extends Module {

  /** Interface for [[Pipe]]s composed of a [[Valid]] input and [[Valid]] output
    * @define notAQueue
    */
  class PipeIO extends Bundle {

    /** [[Valid]] input */
    val enq = Input(Valid(gen))

    /** [[Valid]] output. Data will appear here `latency` cycles after being valid at `enq`. */
    val deq = Output(Valid(gen))
  }

  val io = IO(new PipeIO)

  io.deq <> Pipe(io.enq, latency)
}
