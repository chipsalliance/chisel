// SPDX-License-Identifier: Apache-2.0

package firrtl.passes

import firrtl.ir._
import firrtl.Transform

@deprecated(
  "This pass is an identity transform. For an equivalent dependency, use firrtl.stage.forms.MinimalHighForm",
  "FIRRTL 1.4.2"
)
object ToWorkingIR extends Pass {
  override def prerequisites = firrtl.stage.Forms.MinimalHighForm
  override def optionalPrerequisites = Seq.empty
  override def optionalPrerequisiteOf =
    (firrtl.stage.Forms.LowFormOptimized.toSet -- firrtl.stage.Forms.MinimalHighForm).toSeq
  override def invalidates(a: Transform) = false
  def run(c:                  Circuit): Circuit = c
}
