// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.plugin

import scala.tools.nsc
import nsc.Global
import nsc.plugins.{Plugin, PluginComponent}
import scala.reflect.internal.util.NoPosition
import scala.collection.mutable

private[plugin] case class ChiselPluginArguments(val skipFiles: mutable.HashSet[String] = mutable.HashSet.empty) {
  def useBundlePluginOpt = "useBundlePlugin"
  def useBundlePluginFullOpt = s"-P:${ChiselPlugin.name}:$useBundlePluginOpt"

  // Annoying because this shouldn't be used by users
  def skipFilePluginOpt = "INTERNALskipFile:"
  def skipFilePluginFullOpt = s"-P:${ChiselPlugin.name}:$skipFilePluginOpt"
}

object ChiselPlugin {
  val name = "chiselplugin"

  // Also logs why the compoennt was not run
  private[plugin] def runComponent(global: Global, arguments: ChiselPluginArguments)(unit: global.CompilationUnit): Boolean = {
    // This plugin doesn't work on Scala 2.11 nor Scala 3. Rather than complicate the sbt build flow,
    // instead we just check the version and if its an early Scala version, the plugin does nothing
    val scalaVersion = scala.util.Properties.versionNumberString.split('.')
    val scalaVersionOk = scalaVersion(0).toInt == 2 && scalaVersion(1).toInt >= 12
    val skipFile = arguments.skipFiles(unit.source.file.path)
    if (scalaVersionOk && !skipFile) {
      true
    } else {
      val reason = if (!scalaVersionOk) {
        s"invalid Scala version '${scala.util.Properties.versionNumberString}'"
      } else {
        s"file skipped via '${arguments.skipFilePluginFullOpt}'"
      }
      // Enable this with scalacOption '-Ylog:chiselbundlephase'
      global.log(s"Skipping BundleComponent on account of $reason.")
      false
    }
  }
}

// The plugin to be run by the Scala compiler during compilation of Chisel code
class ChiselPlugin(val global: Global) extends Plugin {
  val name = ChiselPlugin.name
  val description = "Plugin for Chisel 3 Hardware Description Language"
  private val arguments = ChiselPluginArguments()
  val components: List[PluginComponent] = List[PluginComponent](
    new ChiselComponent(global, arguments),
    new BundleComponent(global, arguments)
  )

  override def init(options: List[String], error: String => Unit): Boolean = {
    for (option <- options) {
      if (option == arguments.useBundlePluginOpt) {
        val msg = s"'${arguments.useBundlePluginFullOpt}' is now default behavior, you can stop using the scalacOption."
        global.reporter.warning(NoPosition, msg)
      } else if (option.startsWith(arguments.skipFilePluginOpt)) {
        val filename = option.stripPrefix(arguments.skipFilePluginOpt)
        arguments.skipFiles += filename
        // Be annoying and warn because users are not supposed to use this
        val msg = s"Option -P:${ChiselPlugin.name}:$option should only be used for internal chisel3 compiler purposes!"
        global.reporter.warning(NoPosition, msg)
      } else {
        error(s"Option not understood: '$option'")
      }
    }
    true
  }


}

