// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.experimental.SourceInfo

private[chisel3] trait DisableIntf { self: Disable =>

  /** Logical not
    *
    * @return invert the logical value of this `Disable`
    * @group Bitwise
    */
  def unary_!(using SourceInfo): Disable = _impl_unary_!
}
