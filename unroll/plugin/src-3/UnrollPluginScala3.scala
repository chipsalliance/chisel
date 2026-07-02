package unroll

import dotty.tools.dotc.plugins._

class UnrollPluginScala3 extends StandardPlugin {
  val name: String = "unroll"
  override val description: String = "Count method calls"
  override def init(options: List[String]): List[PluginPhase] = List(new UnrollPhaseScala3())
}