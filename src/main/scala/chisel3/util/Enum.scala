// See LICENSE for license details.

/** Enum generators, allowing circuit constants to have more meaningful names.
  */

package chisel3.util

import chisel3._
import chisel3.internal.chiselRuntimeDeprecated

/** Defines a set of unique UInt constants
  *
  * Unpack with a list to specify an enumeration. Usually used with [[switch]] to describe a finite
  * state machine.
  *
  * @example {{{
  * val state_on :: state_off :: Nil = Enum(2)
  * val current_state = WireInit(state_off)
  * switch (current_state) {
  *   is (state_on) {
  *     ...
  *   }
  *   is (state_off) {
  *     ...
  *   }
  * }
  * }}}
  */
trait Enum {
  /** Returns a sequence of Bits subtypes with values from 0 until n. Helper method. */
  protected def createValues(n: Int): Seq[UInt] =
    (0 until n).map(_.U((1 max log2Ceil(n)).W))

  /** Returns n unique UInt values
    *
    * @param n Number of unique UInt constants to enumerate
    * @return Enumerated constants
    */
  def apply(n: Int): List[UInt] = createValues(n).toList
}

object Enum extends Enum {
  @chiselRuntimeDeprecated
  @deprecated("use Enum(n)", "chisel3, will be removed soon")
  def apply[T <: Bits](nodeType: T, n: Int): List[T] = {
    require(nodeType.isInstanceOf[UInt], "Only UInt supported for enums")
    require(!nodeType.widthKnown, "Bit width may no longer be specified for enums")
    apply(n).asInstanceOf[List[T]]
  }
}
