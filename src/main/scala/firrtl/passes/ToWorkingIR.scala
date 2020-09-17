// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._
import firrtl.Transform

object ToWorkingIR extends Pass {
  override def prerequisites = firrtl.stage.Forms.MinimalHighForm
  override def invalidates(a: Transform) = false
  def run(c:                  Circuit): Circuit = c
}
