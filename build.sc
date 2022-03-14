// SPDX-License-Identifier: Apache-2.0

import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import mill.modules.Util
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:$MILL_VERSION`
import mill.contrib.buildinfo.BuildInfo

import java.io.IOException

object firrtl extends mill.Cross[firrtlCrossModule]("2.12.15", "2.13.7")

class firrtlCrossModule(val crossScalaVersion: String)
    extends CrossSbtModule
    with ScalafmtModule
    with PublishModule
    with BuildInfo {
  override def millSourcePath = super.millSourcePath / os.up

  // 2.12.12 -> Array("2", "12", "12") -> "12" -> 12
  private def majorVersion = crossScalaVersion.split('.')(1).toInt

  def publishVersion = "1.6-SNAPSHOT"

  override def mainClass = T {
    Some("firrtl.stage.FirrtlMain")
  }

  private def javacCrossOptions = Seq("-source", "1.8", "-target", "1.8")

  override def scalacOptions = T {
    super.scalacOptions() ++ Seq(
      "-deprecation",
      "-unchecked",
      "-Yrangepos" // required by SemanticDB compiler plugin
    ) ++ (if (majorVersion == 13) Seq("-Ymacro-annotations") else Nil)
  }

  override def javacOptions = T {
    super.javacOptions() ++ javacCrossOptions
  }

  override def ivyDeps = T {
    super.ivyDeps() ++ Agg(
      ivy"${scalaOrganization()}:scala-reflect:${scalaVersion()}",
      ivy"com.github.scopt::scopt:3.7.1",
      ivy"net.jcazevedo::moultingyaml:0.4.2",
      ivy"org.json4s::json4s-native:3.6.12",
      ivy"org.apache.commons:commons-text:1.9",
      ivy"io.github.alexarchambault::data-class:0.2.5",
      ivy"org.antlr:antlr4-runtime:$antlr4Version",
      ivy"com.google.protobuf:protobuf-java:$protocVersion",
      ivy"com.lihaoyi::os-lib:0.8.1"
    ) ++ {
      if (majorVersion == 13)
        Agg(ivy"org.scala-lang.modules::scala-parallel-collections:1.0.4")
      else
        Agg()
    }
  }

  override def scalacPluginIvyDeps =
    if (majorVersion == 12) Agg(ivy"org.scalamacros:::paradise:2.1.1") else super.scalacPluginIvyDeps

  object test extends Tests {
    override def ivyDeps = T {
      Agg(
        ivy"org.scalatest::scalatest:3.2.10",
        ivy"org.scalatestplus::scalacheck-1-15:3.2.10.0"
      )
    }

    def testFrameworks = T {
      Seq("org.scalatest.tools.Framework")
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
  def antlr4Version = "4.9.3"

  def antlrSource = T.source {
    millSourcePath / "src" / "main" / "antlr4" / "FIRRTL.g4"
  }

  /** override this to false will force FIRRTL using system protoc. */
  val checkSystemAntlr4Version: Boolean = true

  def antlr4Path = T.persistent {
    // Linux distro package antlr4 as antlr4, while brew package as antlr
    PathRef(Seq("antlr4", "antlr").flatMap { f =>
      try {
        val systemAntlr4Version = os.proc(f).call(check = false).out.lines.head.split(" ").last
        if (systemAntlr4Version == antlr4Version || !checkSystemAntlr4Version)
          Some(os.Path(os.proc("bash", "-c", s"command -v $f").call().out.lines.head))
        else
          None
      } catch {
        case _: IOException =>
          None
      }
    }.headOption match {
      case Some(bin) =>
        println(s"Use system antlr4: $bin")
        bin
      case None =>
        println("Download antlr4 from Internet")
        if (!os.isFile(T.ctx.dest / "antlr4"))
          Util.download(s"https://www.antlr.org/download/antlr-$antlr4Version-complete.jar", os.rel / "antlr4.jar")
        T.ctx.dest / "antlr4.jar"
    })
  }

  def generatedAntlr4Source = T.sources {
    antlr4Path().path match {
      case f if f.last == "antlr4.jar" =>
        os.proc(
          "java",
          "-jar",
          f.toString,
          "-o",
          T.ctx.dest.toString,
          "-lib",
          antlrSource().path.toString,
          "-package",
          "firrtl.antlr",
          "-listener",
          "-visitor",
          antlrSource().path.toString
        ).call()
      case _ =>
        os.proc(
          antlr4Path().path.toString,
          "-o",
          T.ctx.dest.toString,
          "-lib",
          antlrSource().path.toString,
          "-package",
          "firrtl.antlr",
          "-listener",
          "-visitor",
          antlrSource().path.toString
        ).call()
    }

    T.ctx.dest
  }

  /* protoc */
  def protocVersion = "3.15.6"

  def protobufSource = T.source {
    millSourcePath / "src" / "main" / "proto" / "firrtl.proto"
  }

  def architecture = T {
    System.getProperty("os.arch")
  }
  def operationSystem = T {
    System.getProperty("os.name")
  }

  /** override this to false will force FIRRTL using system protoc. */
  val checkSystemProtocVersion: Boolean = true
  // For Mac users:
  // MacOS ARM 64-bit supports native binaries via brew:
  // you can install protoc via `brew install protobuf`,
  // If you don't use brew installed protoc
  // It still supports x86_64 binaries via Rosetta 2
  // install via `/usr/sbin/softwareupdate --install-rosetta --agree-to-license`.
  def protocPath = T.persistent {
    PathRef({
      try {
        val systemProtocVersion = os.proc("protoc", "--version").call(check = false).out.lines.head.split(" ").last
        if (systemProtocVersion == protocVersion || !checkSystemProtocVersion)
          Some(os.Path(os.proc("bash", "-c", "command -v protoc").call().out.lines.head))
        else
          None
      } catch {
        case _: IOException =>
          None
      }
    } match {
      case Some(bin) =>
        println(s"Use system protoc: $bin")
        bin
      case None =>
        println("Download protoc from Internet")
        val isMac = operationSystem().toLowerCase.startsWith("mac")
        val isLinux = operationSystem().toLowerCase.startsWith("linux")
        val isWindows = operationSystem().toLowerCase.startsWith("win")

        val aarch_64 = architecture().equals("aarch64") | architecture().startsWith("armv8")
        val ppcle_64 = architecture().equals("ppc64le")
        val s390x = architecture().equals("s390x")
        val x86_32 = architecture().matches("^(x8632|x86|i[3-6]86|ia32|x32)$")
        val x86_64 = architecture().matches("^(x8664|amd64|ia32e|em64t|x64|x86_64)$")

        val protocBinary =
          if (isMac)
            if (aarch_64 || x86_64) "osx-x86_64"
            else throw new Exception("mill cannot detect your architecture of your Mac")
          else if (isLinux)
            if (aarch_64) "linux-aarch_64"
            else if (ppcle_64) "linux-ppcle_64"
            else if (s390x) "linux-s390x"
            else if (x86_32) "linux-x86_32"
            else if (x86_64) "linux-x86_64"
            else throw new Exception("mill cannot detect your architecture of your Linux")
          else if (isWindows)
            if (x86_32) "win32"
            else if (x86_64) "win64"
            else throw new Exception("mill cannot detect your architecture of your Windows")
          else throw new Exception("mill cannot detect your operation system.")

        val unpackPath = os.rel / "unpacked"

        val bin =
          if (isWindows)
            T.ctx.dest / unpackPath / "bin" / "protoc.exe"
          else
            T.ctx.dest / unpackPath / "bin" / "protoc"

        if (!os.exists(bin)) {
          Util.downloadUnpackZip(
            s"https://github.com/protocolbuffers/protobuf/releases/download/v$protocVersion/protoc-$protocVersion-$protocBinary.zip",
            unpackPath
          )
        }
        // Download Linux/Mac binary doesn't have x.
        if (!isWindows) os.perms.set(bin, "rwx------")
        bin
    })
  }

  def generatedProtoSources = T.sources {
    os.proc(
      protocPath().path.toString,
      "-I",
      protobufSource().path / os.up,
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
