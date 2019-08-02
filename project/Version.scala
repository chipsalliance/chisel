// See LICENSE for license details.

sealed trait Branch { def serialize: String }

case class SemanticVersion(major: Int, minor: Int, patch: Int) extends Branch {
  def serialize: String = s"v$major.$minor.$patch"
}

case object Master extends Branch {
  def serialize: String = "master"
}

sealed trait Repository { def serialize: String }

case class GitHub(organization: String, repo: String) extends Repository {
  def serialize: String = s"https://github.com/$organization/$repo"
}

object Version {

  val versionMap: Map[String, (Repository, SemanticVersion)] = Map(
    "chisel3"        -> (GitHub("freechipsproject", "chisel3"),        SemanticVersion(3, 1, 7)),
    "chisel-testers" -> (GitHub("freechipsproject", "chisel-testers"), SemanticVersion(1, 2, 9)),
    "firrtl"         -> (GitHub("freechipsproject", "firrtl"),         SemanticVersion(1, 1, 7)),
    "treadle"        -> (GitHub("freechipsproject", "treadle"),        SemanticVersion(1, 0, 5)) )

  def docSourceUrl(project: String): Seq[String] = {
    val repo = versionMap(project)._1.serialize
    val branch = versionMap(project)._2.serialize
    Seq(
      "-doc-source-url",
      s"$repo/tree/$branch/€{FILE_PATH}.scala" )
  }

}
