// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.tools.nsc
import nsc.Global
import nsc.plugins.{Plugin, PluginComponent}

private[plugin] case class ChiselPluginArguments(var useBundlePlugin: Boolean = false) {
  def useBundlePluginOpt = "useBundlePlugin"
  def useBundlePluginFullOpt = s"-P:chiselplugin:$useBundlePluginOpt"
}

// The plugin to be run by the Scala compiler during compilation of Chisel code
class ChiselPlugin(val global: Global) extends Plugin {
  val name = "chiselplugin"
  val description = "Plugin for Chisel 3 Hardware Description Language"
  private val arguments = ChiselPluginArguments()
  val components: List[PluginComponent] = List[PluginComponent](
    new ChiselComponent(global),
    new BundleComponent(global, arguments)
  )

  override def init(options: List[String], error: String => Unit): Boolean = {
    for (option <- options) {
      if (option == arguments.useBundlePluginOpt) {
        arguments.useBundlePlugin = true
      } else {
        error(s"Option not understood: '$option'")
      }
    }
    true
  }


}

