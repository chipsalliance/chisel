package firrtl
package passes

import scala.collection.mutable
import firrtl.Mappers._
import firrtl.ir._

object RemoveEmpty extends Pass {
  def name = "Remove Empty Statements"
  private def onModule(m: DefModule): DefModule = {
    m match {
      case m: Module => Module(m.info, m.name, m.ports, Utils.squashEmpty(m.body))
      case m: ExtModule => m
    }
  }
  def run(c: Circuit): Circuit = Circuit(c.info, c.modules.map(onModule _), c.main)
}
