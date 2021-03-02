// SPDX-License-Identifier: Apache-2.0

package firrtl.stage.transforms

import firrtl.{BuildInfo, CircuitState, DependencyAPIMigration, Transform}
import firrtl.stage.WarnNoScalaVersionDeprecation
import firrtl.options.StageUtils.dramaticWarning

@deprecated("Support for 2.11 has been dropped, this logic no longer does anything", "FIRRTL 1.5")
object CheckScalaVersion {
  def migrationDocumentLink: String = "https://www.chisel-lang.org/chisel3/upgrading-from-scala-2-11.html"

  final def deprecationMessage(version: String, option: String) =
    s"""|FIRRTL support for Scala $version is deprecated, please upgrade to Scala 2.12.
        |  Migration guide: $migrationDocumentLink
        |  Suppress warning with '$option'""".stripMargin

}

@deprecated("Support for 2.11 has been dropped, this transform no longer does anything", "FIRRTL 1.5")
class CheckScalaVersion extends Transform with DependencyAPIMigration {

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = state
}
