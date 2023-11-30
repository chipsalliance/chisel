// See LICENSE for license details.

import cats.effect.{IO, IOApp}
import org.http4s.client.{Client, JavaNetClientBuilder}
import github4s.{GHResponse, Github}
import github4s.Github._
import github4s.domain.{Pagination, Release}

import cats.effect.unsafe.implicits.global

import Version._

object Releases {

  // These are hardcoded but could be turned into parameters
  val repo = GitHubRepository("chipsalliance", "chisel")
  // Oldest release to fetch
  val oldestRelease = "v3.5.0"

  val httpClient: Client[IO] = JavaNetClientBuilder[IO].create
  val token:      Option[String] = sys.env.get("GITHUB_TOKEN")

  def liftToIO[A](response: GHResponse[A]): IO[A] = response.result match {
    case Left(e) =>
      IO.raiseError(
        new Exception(
          s"Unable to fetch contributors for ${repo.serialize}. Did you misspell it? Did the repository move?" +
            s" Is access token defined: ${token.isDefined}? Original exception: ${e.getMessage}"
        )
      )
    case Right(r) => IO(r)
  }

  // The listReleases API grabs by page so we need to fetch multiple pages
  def getReleases(github: Github[IO], page: Int = 0): IO[List[String]] = for {
    response <- github.repos.listReleases(repo.owner, repo.repo, Some(Pagination(page, 40)), Map())
    fetched <- liftToIO(response).map(_.map(_.tag_name))
    fetchedMore <-
      if (fetched.contains(oldestRelease)) IO(Nil)
      else getReleases(github, page + 1)
  } yield fetched ++ fetchedMore

  def releases(): List[String] =
    getReleases(Github[IO](httpClient, token))
      .unsafeRunSync()

  /* Get latest non-prerelease version */
  def getLatest(releases: List[SemanticVersion]): SemanticVersion = releases.filterNot(_.prerelease).max

  /* Get latest for each major version (newer than v3.5.0)
   *
   * Will pick a non-prerelease if one is available, the newest prerelease if not
   */
  def getLatestForEachMajorVersion(releases: List[SemanticVersion]): List[SemanticVersion] = {
    val oldest = SemanticVersion(3, 5, 0, None, None)
    val filtered = releases.filter(_ > oldest)
    // Get major version but cognizant of pre-5.0 versioning
    def getMajorVersion(v: SemanticVersion): String = {
      if (v.major == 3) s"${v.major}.${v.minor}"
      else v.major.toString
    }
    val grouped = filtered.groupBy(getMajorVersion).map { case (_, values) => values.max }
    grouped.toList.sorted
  }

  def javadocIO(version: SemanticVersion): String =
    if (version.major == 3) {
      s"https://javadoc.io/doc/edu.berkeley.cs/chisel3_2.13/${version.serialize}"
    } else {
      s"https://javadoc.io/doc/org.chipsalliance/chisel_2.13/${version.serialize}"
    }

  def sonatype(version: String): String = {
    val base =
      "https://s01.oss.sonatype.org/service/local/repositories/snapshots/archive/org/chipsalliance/chisel_2.13/"
    s"${base}${version}/chisel_2.13-$version-javadoc.jar/!/index.html"
  }

  def generateMarkdown(snapshot: String): String = {
    val raw = releases()
    val parsed = raw.map(SemanticVersion.parse)
    val latest = getLatest(parsed)
    val major = getLatestForEachMajorVersion(parsed)
    (List(
      s"- [Latest](${javadocIO(latest)})",
      s"- [Snapshot](${sonatype(snapshot)})"
    ) ++ major.reverse.map { v =>
      s"- [${v.serialize}](${javadocIO(v)})"
    }).mkString("\n")

  }
}
