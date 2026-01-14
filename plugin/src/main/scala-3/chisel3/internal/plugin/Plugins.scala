// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import dotty.tools.dotc.plugins.{PluginPhase, StandardPlugin}

class ChiselPlugins extends StandardPlugin {
  val name: String = "ChiselPlugins"
  override val description = "Custom transforms for the Chisel language"
  override def init(options: List[String]): List[PluginPhase] = {
    (new ChiselNamingPhase) :: (new ChiselBundlePhase) :: (new ChiselSourceLocatorPhase) :: Nil
  }
}
