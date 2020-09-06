// See LICENSE for license details.

package firrtl.stage.transforms

import firrtl.{BuildInfo, CircuitState, DependencyAPIMigration, Transform}
import firrtl.stage.WarnNoScalaVersionDeprecation
import firrtl.options.StageUtils.dramaticWarning

object CheckScalaVersion {
  def migrationDocumentLink: String = "https://www.chisel-lang.org/chisel3/upgrading-from-scala-2-11.html"

  private def getScalaMajorVersion: Int = {
    val "2" :: major :: _ :: Nil = BuildInfo.scalaVersion.split("\\.").toList
    major.toInt
  }

  final def deprecationMessage(version: String, option: String) =
    s"""|FIRRTL support for Scala $version is deprecated, please upgrade to Scala 2.12.
        |  Migration guide: $migrationDocumentLink
        |  Suppress warning with '$option'""".stripMargin

}

class CheckScalaVersion extends Transform with DependencyAPIMigration {
  import CheckScalaVersion._

  override def invalidates(a: Transform) = false

  def execute(state: CircuitState): CircuitState = {
    def suppress = state.annotations.contains(WarnNoScalaVersionDeprecation)
    if (getScalaMajorVersion == 11 && !suppress) {
      val option = s"--${WarnNoScalaVersionDeprecation.longOption}"
      dramaticWarning(deprecationMessage("2.11", option))
    }
    state
  }
}
