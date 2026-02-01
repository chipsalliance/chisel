// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait withoutIO$Intf { self: withoutIO.type =>

  /** Creates a new scope in which IO creation causes a runtime error
    *
    * @param block the block of code to run with where IO creation is illegal
    * @return the result of the block
    */
  def apply[T](block: => T)(implicit si: SourceInfo): T = _applyImpl(block)
}
