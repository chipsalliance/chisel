// SPDX-License-Identifier: Apache-2.0

/** Wrappers for valid interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.prefix
import chisel3.util.simpleClassName

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
  * @groupdesc Signals The actual hardware fields of the Bundle
  */
class Valid[+T <: Data](gen: () => T) extends Bundle {

  @deprecated(
    "Use companion object apply to make a Valid. Use constructor that takes () => T if extending Valid.",
    "Chisel 7.9.0"
  )
  def this(gen: T) = this(() => gen)

  /** A bit that will be asserted when `bits` is valid
    * @group Signals
    */
  val valid = Output(Bool())

  /** The data to be transferred, qualified by `valid`
    * @group Signals
    */
  val bits = Output(gen())

  /** True when `valid` is asserted
    * @return a Chisel [[Bool]] true if `valid` is asserted
    */
  def fire: Bool = valid

  /** A non-ambiguous name of this `Valid` instance for use in generated Verilog names
    * Inserts the parameterized generator's typeName, e.g. Valid_UInt4
    */
  override def typeName = s"${simpleClassName(this.getClass)}_${bits.typeName}"

  /** Applies the supplied functor to the bits of this interface, returning a new typed Valid interface.
    * @param f The function to apply to this Valid's 'bits' with return type B
    * @return a new Valid of type B
    */
  def map[B <: Data](f: T => B): Valid[B] = {
    val _map_bits = f(bits)
    val _map = Wire(Valid(chiselTypeOf(_map_bits)))
    _map.bits := _map_bits
    _map.valid := valid
    _map.readOnly
  }
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
  * To convert this to a `valid` interface, you wrap it with a call to the `Valid` companion object's apply method:
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
  * In addition to adding the `valid` bit, a `Valid.fire` method is also added that returns the `valid` bit. This
  *
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
  def apply[T <: Data](gen: T): Valid[T] = new Valid(() => gen)
}
