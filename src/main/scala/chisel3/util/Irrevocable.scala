// SPDX-License-Identifier: Apache-2.0

package chisel3.util

import chisel3._
import chisel3.experimental.Direction
import chisel3.reflect.DataMirror

/** A concrete subclass of ReadyValidIO that promises to not change
  * the value of 'bits' after a cycle where 'valid' is high and 'ready' is low.
  * Additionally, once 'valid' is raised it will never be lowered until after
  * 'ready' has also been raised.
  * @param gen the type of data to be wrapped in IrrevocableIO
  * @groupdesc Signals The actual hardware fields of the Bundle
  */
class IrrevocableIO[+T <: Data](gen: () => T) extends ReadyValidIO[T](gen) {

  @deprecated(
    "Use companion object apply to make an Irrevocable. Use constructor that takes () => T if extending Irrevocable.",
    "Chisel 7.9.0"
  )
  def this(gen: T) = this(() => gen)
}

/** Factory adds an irrevocable handshaking protocol to a data bundle. */
object Irrevocable {
  def apply[T <: Data](gen: T): IrrevocableIO[T] = new IrrevocableIO(() => gen)

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
    val i = Wire(new IrrevocableIO(() => chiselTypeOf(dec.bits)))
    dec.bits := i.bits
    dec.valid := i.valid
    i.ready := dec.ready
    i
  }
}
