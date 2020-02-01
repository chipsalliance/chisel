package firrtl.passes

import firrtl.ir._
import firrtl.Mappers._
import firrtl.options.{PreservesAll}
import firrtl.{Transform, UnknownFlow, UnknownKind, WDefInstance, WRef, WSubAccess, WSubField, WSubIndex}

object ToWorkingIR extends Pass with PreservesAll[Transform] {
  override def prerequisites = firrtl.stage.Forms.MinimalHighForm
  def run(c:Circuit): Circuit = c
}
