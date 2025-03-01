// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.experimental.SourceInfo

/** API for handling disabling of simulation constructs
  *
  * Disables may be non-synthesizable so they can only be used for disabling simulation constructs
  *
  * The default disable is the "hasBeenReset" of the currently in scope reset.
  * It can be set by the user via the [[withDisable]] API
  *
  * Users can access the current `Disable` with [[Module.disable]]
  */
// We could just an OpaqueType, but since OpaqueTypes have some API holes, this non-Data type
class Disable private[chisel3] (private[chisel3] val value: Bool) extends DisableImpl {

  /** Logical not
    *
    * @return invert the logical value of this `Disable`
    * @group Bitwise
    */
  def unary_!(using SourceInfo): Disable = new Disable(!this.value)
}

object Disable extends ObectDisableImpl
