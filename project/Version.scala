// See LICENSE for license details.

import scala.util.control.NonFatal

object Version {

  sealed trait Branch { def serialize: String }

  case class SemanticVersion(major: Int, minor: Int, patch: Int, candidate: Option[Int], milestone: Option[Int])
      extends Branch
      with Ordered[SemanticVersion] {
    require(!(candidate.nonEmpty && milestone.nonEmpty), s"Cannot be both a release candidate and a milestone! $this")

    def prerelease: Boolean = candidate.nonEmpty || milestone.nonEmpty

    private def suffix: String = (candidate.map("-RC" + _).toSeq ++ milestone.map("-M" + _)).mkString

    def serialize: String = s"$major.$minor.$patch" + suffix

    def compare(that: SemanticVersion): Int = SemanticVersion.ordering.compare(this, that)
  }

  object SemanticVersion {

    implicit val ordering: Ordering[SemanticVersion] = {
      // We need None to be greater than Some, default is reversed
      implicit val forOption: Ordering[Option[Int]] = new Ordering[Option[Int]] {
        def compare(x: Option[Int], y: Option[Int]): Int = (x, y) match {
          case (None, None)       => 0
          case (None, _)          => 1
          case (_, None)          => -1
          case (Some(x), Some(y)) => x.compareTo(y)
        }
      }
      Ordering.by(x => (x.major, x.minor, x.patch, x.candidate, x.milestone))
    }

    private val Parsed = """^v(\d+)\.(\d+)\.(\d+)(?:-RC(\d+))?(?:-M(\d+))?""".r

    def parse(str: String): SemanticVersion = try {
      str match {
        case Parsed(major, minor, patch, rc, m) =>
          val rcOpt = Option(rc).map(_.toInt)
          val mOpt = Option(m).map(_.toInt)
          SemanticVersion(major.toInt, minor.toInt, patch.toInt, rcOpt, mOpt)
      }
    } catch {
      case NonFatal(e) =>
        throw new Exception(s"Cannot parse $str as a semantic version, error: $e")
    }
  }

  case object Master extends Branch {
    def serialize: String = "master"
  }

  sealed trait Repository {
    def serialize: String
    def url:       String
  }

  case class GitHubRepository(owner: String, repo: String) extends Repository {
    def serialize: String = s"github.com:$owner/$repo"
    def url:       String = s"https://$serialize"
  }

}
