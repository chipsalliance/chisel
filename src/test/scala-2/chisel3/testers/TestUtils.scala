// SPDX-License-Identifier: Apache-2.0

package chisel3.testers

import chisel3.experimental.SourceInfo
import chisel3.internal.{Builder, Warning, WarningID}

object TestUtils {

  /** Helper for checking warnings, not really valid in normal Chisel */
  def warn(id: Int, msg: String)(implicit sourceInfo: SourceInfo): Unit = Builder.warning(Warning(WarningID(1), msg))

}
