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
    url = "https://github.com/freechipsproject/firrtl.git",
    licenses = Seq(License.`BSD-3-Clause`),
    versionControl = VersionControl.github("freechipsproject", "firrtl"),
    developers = Seq(
      Developer("jackbackrack",    "Jonathan Bachrach",      "https://eecs.berkeley.edu/~jrb/")
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

// Generic antlr4 configuration.
// This could be simpler, but I'm trying to keep some compatibility with the sbt plugin.
case class Antlr4Config(val sourcePath: Path) {
  val ANTLR4_JAR = (home / 'lib / "antlr-4.7.1-complete.jar").toString
  val antlr4GenVisitor: Boolean = true
  val antlr4GenListener: Boolean = false
  val antlr4PackageName: Option[String] = Some("firrtl.antlr")
  val antlr4Version: String = "4.7"

  val listenerArg: String = if (antlr4GenListener) "-listener" else "-no-listener"
  val visitorArg: String = if (antlr4GenVisitor) "-visitor" else "-no-visitor"
  val packageArg: Seq[String] = antlr4PackageName match {
    case Some(p) => Seq("-package", p)
    case None => Seq.empty
  }
  def runAntlr(outputPath: Path) = {
    val cmd = Seq[String]("java", "-jar", ANTLR4_JAR, "-o", outputPath.toString, "-lib", sourcePath.toString, listenerArg, visitorArg) ++ packageArg :+ (sourcePath / "FIRRTL.g4").toString
    val result = %%(cmd)
  }
}

val crossVersions = Seq("2.11.12", "2.12.4")

// Make this available to external tools.
object firrtl extends Cross[FirrtlModule](crossVersions: _*) {
  def defaultVersion(ev: Evaluator[Any]) = T.command{
    println(crossVersions.head)
  }

  def compile = T{
    firrtl(crossVersions.head).compile()
  }

  def jar = T{
    firrtl(crossVersions.head).jar()
  }

  def test = T{
    firrtl(crossVersions.head).test.test()
  }

  def publishLocal = T{
    firrtl(crossVersions.head).publishLocal()
  }

  def docJar = T{
    firrtl(crossVersions.head).docJar()
  }
}

class FirrtlModule(val crossScalaVersion: String) extends CommonModule {
  override def artifactName = "firrtl"

  override def ivyDeps = Agg(
    ivy"com.typesafe.scala-logging::scala-logging:3.7.2",
    ivy"ch.qos.logback:logback-classic:1.2.3",
    ivy"com.github.scopt::scopt:3.6.0",
    ivy"net.jcazevedo::moultingyaml:0.4.0",
    ivy"org.json4s::json4s-native:3.5.3",
    ivy"org.antlr:antlr4-runtime:4.7"
  )

  object test extends Tests {
    override def ivyDeps = Agg(
      ivy"org.scalatest::scalatest:3.0.1",
      ivy"org.scalacheck::scalacheck:1.13.4"
    )
    def testFrameworks = Seq("org.scalatest.tools.Framework")
  }

  def generateAntlrSources(p: Path, sourcePath: Path) = {
    val antlr = new Antlr4Config(sourcePath)
    antlr.runAntlr(p)
    p
  }

  def antlrSourceRoot = T.sources{ pwd / 'src / 'main / 'antlr4 }

  override def generatedSources = T {
    val sourcePath: Path = antlrSourceRoot().head.path
    val p = Seq(PathRef(generateAntlrSources(T.ctx().dest, sourcePath)))
    p
  }
}
