// See LICENSE for license details.

/** Scala-land math helper functions, like logs.
  */

package chisel3.util

import chisel3._
import chisel3.internal.chiselRuntimeDeprecated

/** Compute the log2 rounded up with min value of 1 */
object log2Up {
  @chiselRuntimeDeprecated
  @deprecated("Use log2Ceil instead", "chisel3")
  def apply(in: BigInt): Int = Chisel.log2Up(in)
}

/** Compute the log2 rounded up */
object log2Ceil {
  def apply(in: BigInt): Int = {
    require(in > 0)
    (in-1).bitLength
  }
  def apply(in: Int): Int = apply(BigInt(in))
}

/** Compute the log2 rounded down with min value of 1 */
object log2Down {
  @chiselRuntimeDeprecated
  @deprecated("Use log2Floor instead", "chisel3")
  def apply(in: BigInt): Int = Chisel.log2Down(in)
}

/** Compute the log2 rounded down */
object log2Floor {
  def apply(in: BigInt): Int = log2Ceil(in) - (if (isPow2(in)) 0 else 1)
  def apply(in: Int): Int = apply(BigInt(in))
}

/** Check if an Integer is a power of 2 */
object isPow2 {
  def apply(in: BigInt): Boolean = in > 0 && ((in & (in-1)) == 0)
  def apply(in: Int): Boolean = apply(BigInt(in))
}
