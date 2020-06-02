// Build script for mill 0.6.0
import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.modules.Util

object firrtl extends mill.Cross[firrtlCrossModule]("2.11.12", "2.12.11")

class firrtlCrossModule(crossVersion: String) extends ScalaModule with SbtModule with PublishModule {
  // different scala version shares same sources
  // mill use foo/2.11.12 foo/2.12.11 as millSourcePath by default
  override def millSourcePath = super.millSourcePath / os.up / os.up

  def scalaVersion = crossVersion

  // 2.12.11 -> Array("2", "12", "10") -> "12" -> 12
  private def majorVersion = crossVersion.split('.')(1).toInt

  def publishVersion = "1.4-SNAPSHOT"

  def antlr4Version = "4.7.1"

  def protocVersion = "3.5.1"

  def mainClass = Some("firrtl.stage.FirrtlMain")
  
  private def scalacCrossOptions = majorVersion match {
    case i if i < 12 => Seq()
    case _ => Seq("-Xsource:2.11")
  }

  private def javacCrossOptions = majorVersion match {
    case i if i < 12 => Seq("-source", "1.7", "-target", "1.7")
    case _ => Seq("-source", "1.8", "-target", "1.8")
  }

  override def scalacOptions = super.scalacOptions() ++ Seq(
    "-deprecation",
    "-unchecked",
    "-Yrangepos", // required by SemanticDB compiler plugin
  ) ++ scalacCrossOptions
  
  override def javacOptions = super.javacOptions() ++ javacCrossOptions

  override def ivyDeps = super.ivyDeps() ++ Agg(
    ivy"${scalaOrganization()}:scala-reflect:${scalaVersion()}",
    ivy"com.github.scopt::scopt:3.7.1",
    ivy"net.jcazevedo::moultingyaml:0.4.2",
    ivy"org.json4s::json4s-native:3.6.8",
    ivy"org.apache.commons:commons-text:1.7",
    ivy"org.antlr:antlr4-runtime:4.7.1",
    ivy"com.google.protobuf:protobuf-java:3.5.1"
  )
  
  object test extends Tests {
    private def ivyCrossDeps = majorVersion match {
      case i if i < 12 => Agg(ivy"junit:junit:4.12")
      case _ => Agg()
    }

    def ivyDeps = Agg(
      ivy"org.scalatest::scalatest:3.1.2",
      ivy"org.scalatestplus::scalacheck-1-14:3.1.1.1"
    ) ++ ivyCrossDeps

    def testFrameworks = Seq("org.scalatest.tools.Framework")

    // a sbt-like testOnly command.
    // for example, mill -i "firrtl[2.12.11].test.testOnly" "firrtlTests.AsyncResetSpec"
    def testOnly(args: String*) = T.command {
      super.runMain("org.scalatest.run", args: _*)
    }
  }

  override def generatedSources = T {
    generatedAntlr4Source() ++ generatedProtoSources()
  }

  /** antlr4 */

  def antlrSource = T.source {
    millSourcePath / 'src / 'main / 'antlr4 / "FIRRTL.g4"
  }

  def downloadAntlr4Jar = T {
    Util.download(s"https://www.antlr.org/download/antlr-$antlr4Version-complete.jar")
  }

  def generatedAntlr4Source = T.sources {
    os.proc("java",
      "-jar", downloadAntlr4Jar().path.toString,
      "-o", T.ctx.dest.toString,
      "-lib", antlrSource().path.toString,
      "-package", "firrtl.antlr",
      "-no-listener", "-visitor",
      antlrSource().path.toString
    ).call()
    T.ctx.dest
  }

  /** protoc */

  def protobufSource = T.source {
    millSourcePath / 'src / 'main / 'proto / "firrtl.proto"
  }

  def downloadProtocJar = T {
    Util.download(s"https://repo.maven.apache.org/maven2/com/github/os72/protoc-jar/$protocVersion/protoc-jar-$protocVersion.jar")
  }

  def generatedProtoSources = T.sources {
    os.proc("java",
      "-jar", downloadProtocJar().path.toString,
      "-I", protobufSource().path / os.up,
      s"--java_out=${T.ctx.dest.toString}",
      protobufSource().path.toString()
    ).call()
    T.ctx.dest / "firrtl"
  }

  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "edu.berkeley.cs",
    url = "https://github.com/freechipsproject/firrtl",
    licenses = Seq(License.`BSD-3-Clause`),
    versionControl = VersionControl.github("freechipsproject", "firrtl"),
    developers = Seq(
      Developer("jackbackrack", "Jonathan Bachrach", "https://eecs.berkeley.edu/~jrb/")
    )
  )
  // make mill publish sbt compatible package
  def artifactName = "firrtl"
}
