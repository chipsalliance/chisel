import $ivy.`com.github.lolgab::mill-mima::0.0.23`
import $ivy.`io.chris-kipp::mill-ci-release::0.1.9`

import mill._
import mill.scalalib._
import mill.scalalib.TestModule._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import coursier.maven.MavenRepository
import mill.scalalib.api.ZincWorkerUtil.matchingVersions
import com.github.lolgab.mill.mima._
import io.kipp.mill.ci.release.{CiReleaseModule, SonatypeHost}
import $file.common

object v {
  val pluginScalaCrossVersions = Seq(
    "2.13.0",
    "2.13.1",
    "2.13.2",
    "2.13.3",
    "2.13.4",
    "2.13.5",
    "2.13.6",
    "2.13.7",
    "2.13.8",
    "2.13.9",
    "2.13.10",
    "2.13.11"
  )
  val scalaCrossVersions = Seq(
    "2.13.11"
  )
  val osLib = ivy"com.lihaoyi::os-lib:0.9.1"
  val upickle = ivy"com.lihaoyi::upickle:3.1.0"
  val scalatest = ivy"org.scalatest::scalatest:3.2.14"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-15:3.2.11.0"
  val json4s = ivy"org.json4s::json4s-native:4.0.6"
  val dataclass = ivy"io.github.alexarchambault::data-class:0.2.5"
  val commonText = ivy"org.apache.commons:commons-text:1.10.0"
  val scopt = ivy"com.github.scopt::scopt:3.7.1"

  def scalaReflect(scalaVersion: String) = ivy"org.scala-lang:scala-reflect:$scalaVersion"

  def scalaCompiler(scalaVersion: String) = ivy"org.scala-lang:scala-compiler:$scalaVersion"

  def scalaLibrary(scalaVersion: String) = ivy"org.scala-lang:scala-library:$scalaVersion"
}

object firrtl extends Cross[Firrtl](v.scalaCrossVersions)

trait Firrtl
    extends common.FirrtlModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "firrtl"

  def osLibModuleIvy = v.osLib

  def json4sIvy = v.json4s

  def dataclassIvy = v.dataclass

  def commonTextIvy = v.commonText

  def scoptIvy = v.scopt
}

object svsim extends Cross[Svsim](v.scalaCrossVersions)

trait Svsim
    extends common.SvsimModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "svsim"
}

object firrtlut extends Cross[FirrtlUnitTest](v.scalaCrossVersions)

trait FirrtlUnitTest
    extends common.FirrtlUnitTestModule
    with CrossModuleBase
    with ScalafmtModule {
  override def millSourcePath = firrtl(crossScalaVersion).millSourcePath

  def firrtlModule = firrtl(crossScalaVersion)

  def scalatestIvy = v.scalatest

  def scalacheckIvy = v.scalacheck

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "src" / "test")) ++
      matchingVersions(crossScalaVersion).map(s => PathRef(millSourcePath / "src" / "test" / s"scala-$s"))
  }
}

object macros extends Cross[Macros](v.scalaCrossVersions)

trait Macros
    extends common.MacrosModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "macros"

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)
}

object core extends Cross[Core](v.scalaCrossVersions)

trait Core
    extends common.CoreModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "core"

  def firrtlModule = firrtl(crossScalaVersion)

  def macrosModule = macros(crossScalaVersion)

  def osLibModuleIvy = v.osLib

  def upickleModuleIvy = v.upickle

  def firtoolVersion = T {
    import scala.sys.process._
    val Version = """^CIRCT firtool-(\S+)$""".r
    try {
      val lines = Process(Seq("firtool", "--version")).lineStream
      lines.collectFirst {
        case Version(v) => Some(v)
        case _          => None
      }.get
    } catch {
      case e: java.io.IOException => None
    }
  }

  def buildVersion = T(os.proc("git", "describe", "--tags", "--dirty").call().out.lines.head.stripPrefix("v"))
}

object plugin extends Cross[Plugin](v.pluginScalaCrossVersions)

trait Plugin
    extends common.PluginModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "plugin"

  def scalaLibraryIvy = v.scalaLibrary(crossScalaVersion)

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)

  def scalaCompilerIvy: Dep = v.scalaCompiler(crossScalaVersion)
}

object chisel extends Cross[Chisel](v.scalaCrossVersions)

trait Chisel
    extends common.ChiselModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  override def millSourcePath = super.millSourcePath / os.up

  def svsimModule = svsim(crossScalaVersion)

  def macrosModule = macros(crossScalaVersion)

  def coreModule = core(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object chiselut extends Cross[ChiselUnitTest](v.scalaCrossVersions)

trait ChiselUnitTest
    extends common.ChiselUnitTestModule
    with CrossModuleBase
    with ScalafmtModule {
  override def millSourcePath = chisel(crossScalaVersion).millSourcePath

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)

  def scalatestIvy = v.scalatest

  def scalacheckIvy = v.scalacheck

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "src" / "test")) ++
      matchingVersions(crossScalaVersion).map(s => PathRef(millSourcePath / "src" / "test" / s"scala-$s"))
  }
}

object stdlib extends Cross[Stdlib](v.scalaCrossVersions)

trait Stdlib
    extends common.StdLibModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "stdlib"

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

trait ChiselPublishModule
    extends PublishModule
    with CiReleaseModule
    with Mima {
  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "org.chipsalliance",
    url = "https://www.chisel-lang.org",
    licenses = Seq(License.`Apache-2.0`),
    versionControl = VersionControl.github("chipsalliance", "chisel"),
    developers = Seq()
  )
  def mimaPreviousVersions = os.read.lines(os.pwd / "project" / "previous-versions.txt")

  override def sonatypeHost = Some(SonatypeHost.s01)
}
