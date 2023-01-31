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

  val token: Option[String] = sys.env.get("GITHUB_TOKEN")

  def contributors(repo: GitHubRepository): List[User] =
    Github[IO](httpClient, token)
      .repos
      .listContributors(repo.owner, repo.repo)
      .unsafeRunSync()
      .result match {
        case Left(e) => throw new Exception(
          s"Unable to fetch contributors for ${repo.serialize}. Did you misspell it? Did the repository move?" +
          s" Is access token defined: ${token.isDefined}? Original exception: ${e.getMessage}"
        )
        case Right(r) => r
      }

  def contributorsMarkdown(contributors: Seq[(String, String)]): String =
    contributors
      .sortBy(_._1.toLowerCase)
      .map { case (login, html_url) => s"- [`@${login}`](${html_url})" }
      .mkString("\n")

}
