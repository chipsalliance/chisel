// See LICENSE for license details.

package firrtl
package passes

import firrtl.ir._
import firrtl.options.PreservesAll
import firrtl.stage.Forms

object RemoveEmpty extends Pass with DependencyAPIMigration with PreservesAll[Transform] {

  override def prerequisites = Seq.empty
  override def optionalPrerequisites = Forms.LowFormOptimized
  override def optionalPrerequisiteOf = Forms.ChirrtlEmitters

  private def onModule(m: DefModule): DefModule = {
    m match {
      case m: Module => Module(m.info, m.name, m.ports, Utils.squashEmpty(m.body))
      case m: ExtModule => m
    }
  }
  def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule), c.main)
}
