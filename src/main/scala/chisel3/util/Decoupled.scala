// SPDX-License-Identifier: Apache-2.0

/** Wrappers for ready-valid (Decoupled) interfaces and associated circuit generators using them.
  */

package chisel3.util

import chisel3._
import chisel3.experimental.Direction
import chisel3.reflect.DataMirror

/** A concrete subclass of ReadyValidIO signaling that the user expects a
  * "decoupled" interface: 'valid' indicates that the producer has
  * put valid data in 'bits', and 'ready' indicates that the consumer is ready
  * to accept the data this cycle. No requirements are placed on the signaling
  * of ready or valid.
  * @param gen the type of data to be wrapped in DecoupledIO
  */
class DecoupledIO[+T <: Data](gen: () => T) extends ReadyValidIO[T](gen) {

  @deprecated(
    "Use companion object apply to make a Decoupled. Use constructor that takes () => T if extending Decoupled.",
    "Chisel 7.9.0"
  )
  def this(gen: T) = this(() => gen)

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
  def apply[T <: Data](gen: T): DecoupledIO[T] = new DecoupledIO(() => gen)

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
    val d = Wire(new DecoupledIO(() => chiselTypeOf(irr.bits)))
    d.bits := irr.bits
    d.valid := irr.valid
    irr.ready := d.ready
    d
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
