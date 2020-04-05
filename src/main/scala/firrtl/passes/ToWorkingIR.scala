package firrtl.passes

import firrtl.ir._
import firrtl.options.PreservesAll
import firrtl.Transform

object ToWorkingIR extends Pass with PreservesAll[Transform] {
  override def prerequisites = firrtl.stage.Forms.MinimalHighForm
  def run(c:Circuit): Circuit = c
}
