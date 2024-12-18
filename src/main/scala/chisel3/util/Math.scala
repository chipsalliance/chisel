// SPDX-License-Identifier: Apache-2.0

/** Scala-land math helper functions, like logs.
  */

package chisel3.util

import chisel3.internal

/** Compute the log2 of a Scala integer, rounded up, with min value of 1.
  * Useful for getting the number of bits needed to represent some number of states (in - 1),
  * To get the number of bits needed to represent some number n, use log2Up(n + 1).
  * with the minimum value preventing the creation of currently-unsupported zero-width wires.
  *
  * Note: prefer to use log2Ceil when in is known to be > 1 (where log2Ceil(in) > 0).
  * This will be deprecated when zero-width wires is supported.
  *
  * @example {{{
  * log2Up(1)  // returns 1
  * log2Up(2)  // returns 1
  * log2Up(3)  // returns 2
  * log2Up(4)  // returns 2
  * }}}
  */
object log2Up {
  // Do not deprecate until zero-width wires fully work:
  // https://github.com/freechipsproject/chisel3/issues/847
  //@chiselRuntimeDeprecated
  //@deprecated("Use log2Ceil instead", "chisel3")
  def apply(in: BigInt): Int = {
    require(in >= 0, s"log2Up is only defined on integers >= 0, got $in")
    1.max((in - 1).bitLength)
  }
  def apply(in: Int): Int = apply(BigInt(in))
}

/** Compute the log2 of a Scala integer, rounded up.
  * Useful for getting the number of bits needed to represent some number of states (in - 1).
  * To get the number of bits needed to represent some number n, use log2Ceil(n + 1).
  *
  * Note: can return zero, and should not be used in cases where it may generate unsupported
  * zero-width wires.
  *
  * @example {{{
  * log2Ceil(1)  // returns 0
  * log2Ceil(2)  // returns 1
  * log2Ceil(3)  // returns 2
  * log2Ceil(4)  // returns 2
  * }}}
  */
object log2Ceil {
  def apply(in: BigInt): Int = {
    require(in > 0, s"log2 is only defined on integers > 0, got $in")
    (in - 1).bitLength
  }
  def apply(in: Int): Int = apply(BigInt(in))
}

/** Compute the log2 of a Scala integer, rounded down, with min value of 1.
  *
  * @example {{{
  * log2Down(1)  // returns 1
  * log2Down(2)  // returns 1
  * log2Down(3)  // returns 1
  * log2Down(4)  // returns 2
  * }}}
  */
object log2Down {
  // Do not deprecate until zero-width wires fully work:
  // https://github.com/freechipsproject/chisel3/issues/847
  //@chiselRuntimeDeprecated
  //@deprecated("Use log2Floor instead", "chisel3")
  def apply(in: BigInt): Int = log2Up(in) - (if (isPow2(in)) 0 else 1)
  def apply(in: Int):    Int = apply(BigInt(in))
}

/** Compute the log2 of a Scala integer, rounded down.
  *
  * Can be useful in computing the next-smallest power of two.
  *
  * @example {{{
  * log2Floor(1)  // returns 0
  * log2Floor(2)  // returns 1
  * log2Floor(3)  // returns 1
  * log2Floor(4)  // returns 2
  * }}}
  */
object log2Floor {
  def apply(in: BigInt): Int = log2Ceil(in) - (if (isPow2(in)) 0 else 1)
  def apply(in: Int):    Int = apply(BigInt(in))
}

/** Returns whether a Scala integer is a power of two.
  *
  * @example {{{
  * isPow2(1)  // returns true
  * isPow2(2)  // returns true
  * isPow2(3)  // returns false
  * isPow2(4)  // returns true
  * }}}
  */
object isPow2 {
  def apply(in: BigInt): Boolean = in > 0 && ((in & (in - 1)) == 0)
  def apply(in: Int):    Boolean = apply(BigInt(in))
}

object unsignedBitLength {

  /** Return the number of bits required to encode a specific value, assuming no sign bit is required.
    *
    * Basically, `n.bitLength`. NOTE: This will return 0 for a value of 0.
    * This reflects the Chisel assumption that a zero width wire has a value of 0.
    * @param in - the number to be encoded.
    * @return - an Int representing the number of bits to encode.
    */
  def apply(in: BigInt): Int = {
    require(in >= 0, s"Unsigned integers must be non-negative, got $in")
    in.bitLength
  }
}

object signedBitLength {

  /** Return the number of bits required to encode a specific value, assuming a sign bit is required.
    *
    * Basically, 0 for 0, 1 for -1, and `n.bitLength` + 1 for everything else.
    * This reflects the Chisel assumption that a zero width wire has a value of 0.
    * @param in - the number to be encoded.
    * @return - an Int representing the number of bits to encode.
    */
  def apply(in: BigInt): Int = {
    in.toInt match {
      case 0  => 0
      case -1 => 1
      case _  => in.bitLength + 1
    }

  }
}
