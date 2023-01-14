// See LICENSE for license details.

object Version {

  sealed trait Branch { def serialize: String }

  case class SemanticVersion(major: Int, minor: Int, patch: Int) extends Branch {
    def serialize: String = s"v$major.$minor.$patch"
  }

  case object Master extends Branch {
    def serialize: String = "master"
  }

  sealed trait Repository {
    def serialize: String
    def url: String
  }

  case class GitHubRepository(owner: String, repo: String) extends Repository {
    def serialize: String = s"github.com:$owner/$repo"
    def url: String = s"https://$serialize"
  }

  val versionMap: Map[String, (Repository, SemanticVersion)] = Map(
    "chisel3"        -> (GitHubRepository("freechipsproject", "chisel3"),        SemanticVersion(3, 1, 7)),

  def docSourceUrl(project: String): Seq[String] = {
    val repo = versionMap(project)._1.url
    val branch = versionMap(project)._2.serialize
    Seq(
      "-doc-source-url",
      s"$repo/tree/$branch/€{FILE_PATH}.scala" )
  }

}
