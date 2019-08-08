// See LICENSE for license details.

import Version.GitHubRepository

object Contributors {

  import github4s.Github
  import github4s.Github._
  import github4s.free.domain.User
  import github4s.jvm.Implicits._
  import scalaj.http.HttpResponse

  val token: Option[String] = sys.env.get("GITHUB4S_ACCESS_TOKEN")

  def contributors(repo: GitHubRepository): List[User] =
    Github(token)
      .repos
      .listContributors(repo.owner, repo.repo)
      .exec[cats.Id, HttpResponse[String]]() match {
        case Left(e) => throw new Exception(s"Unable to fetch contributors for ${repo.serialize}. Did you misspell it?")
        case Right(r) => r.result
      }

  def contributorsMarkdown(contributors: Seq[User]): String =
    contributors
      .sortWith((a: User, b: User) => a.login.toLowerCase < b.login.toLowerCase)
      .map(a => s"- [`@${a.login}`](${a.html_url})")
      .mkString("\n")

}
