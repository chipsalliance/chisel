// SPDX-License-Identifier: Apache-2.0

package firrtl.transforms

object FixAddingNegativeLiterals {

  /** Returns the maximum negative number represented by given width
    * @param width width of the negative number
    * @return maximum negative number
    */
  def minNegValue(width: BigInt): BigInt = -(BigInt(1) << (width.toInt - 1))

}
