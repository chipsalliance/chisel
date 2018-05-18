import ammonite.ops._
import ammonite.ops.ImplicitWd._
import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.eval.Evaluator

import $file.CommonBuild

// An sbt layout with src in the top directory.
trait CrossUnRootedSbtModule extends CrossSbtModule {
  override def millSourcePath = super.millSourcePath / ammonite.ops.up
}

trait CommonModule extends CrossUnRootedSbtModule with PublishModule {
  def publishVersion = "1.2-SNAPSHOT"

  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "edu.berkeley.cs",
    url = "https://github.com/freechipsproject/firrtl-interpreter.git",
    licenses = Seq(License.`BSD-3-Clause`),
    versionControl = VersionControl.github("freechipsproject", "firrtl-interpreter"),
    developers = Seq(
      Developer("chick",    "Charles Markley",      "https://aspire.eecs.berkeley.edu/author/chick/")
    )
  )

  override def scalacOptions = Seq(
    "-deprecation",
    "-explaintypes",
    "-feature", "-language:reflectiveCalls",
    "-unchecked",
    "-Xcheckinit",
    "-Xlint:infer-any",
    "-Xlint:missing-interpolator"
  ) ++ CommonBuild.scalacOptionsVersion(crossScalaVersion)

  override def javacOptions = CommonBuild.javacOptionsVersion(crossScalaVersion)
}

val crossVersions = Seq("2.11.12", "2.12.4")

// Make this available to external tools.
object firrtlInterpreter extends Cross[FirrtlInterpreterModule](crossVersions: _*) {
  def defaultVersion(ev: Evaluator[Any]) = T.command{
    println(crossVersions.head)
  }

  def compile = T{
    firrtlInterpreter(crossVersions.head).compile()
  }

  def jar = T{
    firrtlInterpreter(crossVersions.head).jar()
  }

  def test = T{
    firrtlInterpreter(crossVersions.head).test.test()
  }

  def publishLocal = T{
    firrtlInterpreter(crossVersions.head).publishLocal()
  }

  def docJar = T{
    firrtlInterpreter(crossVersions.head).docJar()
  }
}

// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map("firrtl" -> "1.2-SNAPSHOT")

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}

class FirrtlInterpreterModule(val crossScalaVersion: String) extends CommonModule {
  override def artifactName = "firrtl-interpreter"

  def chiselDeps = Agg("firrtl").map { d => getVersion(d) }

  override def ivyDeps = Agg(
    ivy"org.scala-lang.modules:scala-jline:2.12.1"
  ) ++ chiselDeps

  object test extends Tests {
    override def ivyDeps = Agg(
      ivy"org.scalatest::scalatest:3.0.1",
      ivy"org.scalacheck::scalacheck:1.13.4"
    )
    def testFrameworks = Seq("org.scalatest.tools.Framework")
  }

}
