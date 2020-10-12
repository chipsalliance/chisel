// SPDX-License-Identifier: Apache-2.0

import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.modules.Util
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:$MILL_VERSION`
import mill.contrib.buildinfo.BuildInfo

object firrtl extends mill.Cross[firrtlCrossModule]("2.11.12", "2.12.12", "2.13.2")

class firrtlCrossModule(val crossScalaVersion: String) extends CrossSbtModule with PublishModule with BuildInfo {
  override def millSourcePath = super.millSourcePath / os.up

  // 2.12.12 -> Array("2", "12", "12") -> "12" -> 12
  private def majorVersion = crossScalaVersion.split('.')(1).toInt

  def publishVersion = "1.4-SNAPSHOT"

  override def mainClass = T {
    Some("firrtl.stage.FirrtlMain")
  }

  private def javacCrossOptions = majorVersion match {
    case i if i < 12 => Seq("-source", "1.7", "-target", "1.7")
    case _ => Seq("-source", "1.8", "-target", "1.8")
  }

  override def scalacOptions = T {
    super.scalacOptions() ++ Seq(
      "-deprecation",
      "-unchecked",
      "-Yrangepos" // required by SemanticDB compiler plugin
    )
  }

  override def javacOptions = T {
    super.javacOptions() ++ javacCrossOptions
  }

  override def ivyDeps = T {
    super.ivyDeps() ++ Agg(
      ivy"${scalaOrganization()}:scala-reflect:${scalaVersion()}",
      ivy"com.github.scopt::scopt:3.7.1",
      ivy"net.jcazevedo::moultingyaml:0.4.2",
      ivy"org.json4s::json4s-native:3.6.9",
      ivy"org.apache.commons:commons-text:1.8",
      ivy"org.antlr:antlr4-runtime:$antlr4Version",
      ivy"com.google.protobuf:protobuf-java:$protocVersion"
    ) ++ {
      if (majorVersion > 12)
        Agg(ivy"org.scala-lang.modules::scala-parallel-collections:0.2.0")
      else
        Agg()
    }
  }

  object test extends Tests {
    private def ivyCrossDeps = majorVersion match {
      case i if i < 12 => Agg(ivy"junit:junit:4.13.1")
      case _ => Agg()
    }

    override def ivyDeps = T {
      Agg(
        ivy"org.scalatest::scalatest:3.2.0",
        ivy"org.scalatestplus::scalacheck-1-14:3.1.3.0"
      ) ++ ivyCrossDeps
    }

    def testFrameworks = T {
      Seq("org.scalatest.tools.Framework")
    }

    // a sbt-like testOnly command.
    // for example, mill -i "firrtl[2.12.12].test.testOnly" "firrtlTests.AsyncResetSpec"
    def testOnly(args: String*) = T.command {
      super.runMain("org.scalatest.run", args: _*)
    }
  }

  override def buildInfoPackageName = Some("firrtl")

  override def buildInfoMembers: T[Map[String, String]] = T {
    Map(
      "buildInfoPackage" -> artifactName(),
      "version" -> publishVersion(),
      "scalaVersion" -> scalaVersion()
    )
  }

  override def generatedSources = T {
    generatedAntlr4Source() ++ generatedProtoSources() :+ generatedBuildInfo()._2
  }

  /* antlr4 */
  def antlr4Version = "4.7.1"

  def antlrSource = T.source {
    millSourcePath / "src" / "main" / "antlr4" / "FIRRTL.g4"
  }

  def downloadAntlr4Jar = T.persistent {
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

  /* protoc */
  def protocVersion = "3.5.1"

  def protobufSource = T.source {
    millSourcePath / "src" / "main" / "proto" / "firrtl.proto"
  }

  def downloadProtocJar = T.persistent {
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

  def pomSettings = T {
    PomSettings(
      description = artifactName(),
      organization = "edu.berkeley.cs",
      url = "https://github.com/freechipsproject/firrtl",
      licenses = Seq(License.`BSD-3-Clause`),
      versionControl = VersionControl.github("freechipsproject", "firrtl"),
      developers = Seq(
        Developer("jackbackrack", "Jonathan Bachrach", "https://eecs.berkeley.edu/~jrb/")
      )
    )
  }

  // make mill publish sbt compatible package
  override def artifactName = "firrtl"
}
