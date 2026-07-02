package unroll

import tools.nsc.Global

class UnrollPluginScala2(val global: Global) extends tools.nsc.plugins.Plugin {
  val name = "unroll"
  val description = "Plugin to unroll default methods for binary compatibility"
  val components = List(new UnrollPhaseScala2(this.global))
}