// See LICENSE for license details.

import Version.GitHubRepository

object Contributors {

  import github4s.Github
  import github4s.Github._
  import github4s.domain.User

  import java.util.concurrent.Executors

  import cats.effect.{Blocker, ContextShift, IO}
  import org.http4s.client.{Client, JavaNetClientBuilder}

  import scala.concurrent.ExecutionContext.global

  val httpClient: Client[IO] = {
    val blockingPool = Executors.newFixedThreadPool(5)
    val blocker = Blocker.liftExecutorService(blockingPool)
    implicit val cs: ContextShift[IO] = IO.contextShift(global)
    JavaNetClientBuilder[IO](blocker).create // use BlazeClientBuilder for production use
  }

  val token: Option[String] = sys.env.get("GITHUB4S_ACCESS_TOKEN")

  def contributors(repo: GitHubRepository): List[User] =
    Github[IO](httpClient, token)
      .repos
      .listContributors(repo.owner, repo.repo)
      .unsafeRunSync()
      .result match {
        case Left(e) => throw new Exception(s"Unable to fetch contributors for ${repo.serialize}. Did you misspell it? Did the repository move?")
        case Right(r) => r
      }

  def contributorsMarkdown(contributors: Seq[User]): String =
    contributors
      .sortWith((a: User, b: User) => a.login.toLowerCase < b.login.toLowerCase)
      .map(a => s"- [`@${a.login}`](${a.html_url})")
      .mkString("\n")

}
