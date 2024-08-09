import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import mill.define.Cross
import mill.scalalib.api.ZincWorkerUtil.matchingVersions
import $ivy.`com.lihaoyi::mill-contrib-jmh:`
import mill.contrib.jmh.JmhModule
import $file.common
import $file.tests

import common.Platform

object v {
  val pluginScalaCrossVersions = Seq(
    "2.13.11",
    "2.13.12",
    "2.13.13",
    "2.13.14"
  )
  val scalaCrossVersions = Seq(
    "2.13.14"
  )
  val scalaVersion = scalaCrossVersions.head
  val jmhVersion = "1.37"
  val osLib = ivy"com.lihaoyi::os-lib:0.10.0"
  val upickle = ivy"com.lihaoyi::upickle:3.3.1"
  val firtoolResolver = ivy"org.chipsalliance::firtool-resolver:2.0.0"
  val scalatest = ivy"org.scalatest::scalatest:3.2.19"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-18:3.2.19.0"
  val json4s = ivy"org.json4s::json4s-native:4.0.7"
  val dataclass = ivy"io.github.alexarchambault::data-class:0.2.6"
  val commonText = ivy"org.apache.commons:commons-text:1.12.0"
  val scopt = ivy"com.github.scopt::scopt:4.1.0"

  def scalaReflect(scalaVersion: String) = ivy"org.scala-lang:scala-reflect:$scalaVersion"

  def scalaCompiler(scalaVersion: String) = ivy"org.scala-lang:scala-compiler:$scalaVersion"

  def scalaLibrary(scalaVersion: String) = ivy"org.scala-lang:scala-library:$scalaVersion"
}

object utils extends Module {
  def firtoolVersion = T {
    val contents = os.read(millSourcePath / os.up / "etc" / "circt.json")
    val read = upickle.default.read[Map[String, String]](contents)
    read("version").stripPrefix("firtool-")
  }
}

object firrtl extends Cross[Firrtl](v.scalaCrossVersions)

trait Firrtl extends common.FirrtlModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "firrtl"

  def osLibModuleIvy = v.osLib

  def json4sIvy = v.json4s

  def dataclassIvy = v.dataclass

  def commonTextIvy = v.commonText

  def scoptIvy = v.scopt
}

object svsim extends Cross[Svsim](v.scalaCrossVersions)

trait Svsim extends common.SvsimModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "svsim"
}

object firrtlut extends Cross[FirrtlUnitTest](v.scalaCrossVersions)

trait FirrtlUnitTest extends tests.FirrtlUnitTestModule with CrossModuleBase with ScalafmtModule {
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

trait Macros extends common.MacrosModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "macros"

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)
}

object core extends Cross[Core](v.scalaCrossVersions)

trait Core extends common.CoreModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "core"

  def firrtlModule = firrtl(crossScalaVersion)

  def macrosModule = macros(crossScalaVersion)

  def osLibModuleIvy = v.osLib

  def upickleModuleIvy = v.upickle

  def firtoolResolverModuleIvy = v.firtoolResolver

  def buildVersion = T("build-from-source")

  private def generateBuildInfo = T {
    val outputFile = T.dest / "chisel3" / "BuildInfo.scala"
    val firtoolVersionString = "Some(\"" + utils.firtoolVersion() + "\")"
    val contents =
      s"""
         |package chisel3
         |case object BuildInfo {
         |  val buildInfoPackage: String = "${artifactName()}"
         |  val version: String = "${buildVersion()}"
         |  val scalaVersion: String = "${scalaVersion()}"
         |  val firtoolVersion: scala.Option[String] = $firtoolVersionString
         |  override val toString: String = {
         |    "buildInfoPackage: %s, version: %s, scalaVersion: %s, firtoolVersion %s".format(
         |        buildInfoPackage, version, scalaVersion, firtoolVersion
         |    )
         |  }
         |}
         |""".stripMargin
    os.write(outputFile, contents, createFolders = true)
    PathRef(T.dest)
  }

  override def generatedSources = T {
    super.generatedSources() :+ generateBuildInfo()
  }
}

object plugin extends Cross[Plugin](v.pluginScalaCrossVersions)

trait Plugin extends common.PluginModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "plugin"

  def scalaLibraryIvy = v.scalaLibrary(crossScalaVersion)

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)

  def scalaCompilerIvy: Dep = v.scalaCompiler(crossScalaVersion)
}

object chisel extends Cross[Chisel](v.scalaCrossVersions)

trait Chisel extends common.ChiselModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  override def millSourcePath = super.millSourcePath / os.up

  def svsimModule = svsim(crossScalaVersion)

  def macrosModule = macros(crossScalaVersion)

  def coreModule = core(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object chiselut extends Cross[ChiselUnitTest](v.scalaCrossVersions)

trait ChiselUnitTest extends tests.ChiselUnitTestModule with CrossModuleBase with ScalafmtModule {
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

trait Stdlib extends common.StdLibModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "stdlib"

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

trait ChiselPublishModule extends PublishModule {
  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "org.chipsalliance",
    url = "https://www.chisel-lang.org",
    licenses = Seq(License.`Apache-2.0`),
    versionControl = VersionControl.github("chipsalliance", "chisel"),
    developers = Seq()
  )

  def publishVersion = "5.0-SNAPSHOT"
}

object jextract extends Module {

  def platformName = Platform.getPlatform() match {
    case Platform.Linux => "linux"
    case Platform.Macos => "macos"
  }

  def downloadUrl =
    s"https://download.java.net/java/early_access/jextract/21/1/openjdk-21-jextract+1-2_$platformName-x64_bin.tar.gz"

  def tarball: T[os.Path] = T {
    def tarballName = "jextract.tar.gz"
    def path = T.dest

    val file = T.dest / tarballName
    os.write(file, requests.get.stream(downloadUrl))
    file
  }

  def extracted: T[os.Path] = T {
    os.makeDir.all(T.dest)
    os.proc("tar", "zxf", tarball(), "--strip-components=1")
      .call(cwd = T.dest)
    T.dest
  }

  def bin: T[os.Path] = T {
    extracted() / "bin" / "jextract"
  }
}

object circt extends Module {

  def platformName = Platform.getPlatform() match {
    case Platform.Linux => "linux"
    case Platform.Macos => "macos"
  }

  def downloadUrl = T {
    s"https://github.com/llvm/circt/releases/download/firtool-${utils.firtoolVersion()}/circt-full-shared-$platformName-x64.tar.gz"
  }

  def tarball: T[os.Path] = T {
    def tarballName = "circt-full-shared.tar.gz"
    def path = T.dest

    val file = T.dest / tarballName
    os.write(file, requests.get.stream(downloadUrl()))
    file
  }

  def extracted: T[os.Path] = T {
    os.makeDir.all(T.dest)
    os.proc("tar", "zxf", tarball(), "--strip-components=1")
      .call(cwd = T.dest)
    T.dest
  }
}

object circtpanamabinding extends CIRCTPanamaBinding {
  def jextractBinary: T[String] = T.input {
    T.ctx().env.get("JEXTRACT_BINARY") match {
      case Some(bin) => bin
      case None      => jextract.bin().toString
    }
  }
}

trait CIRCTPanamaBinding extends common.CIRCTPanamaBindingModule with ChiselPublishModule {

  def header = T(PathRef(millSourcePath / "jextract-headers.h"))

  def circtInstallPath = T.input {
    T.ctx().env.get("CIRCT_INSTALL_PATH") match {
      case Some(dir) => os.Path(dir)
      case None      => circt.extracted()
    }
  }

  def includePaths = T(Seq(PathRef(circtInstallPath() / "include")))

  def libraryPaths = T(Seq(PathRef(circtInstallPath() / "lib")))
}

object panamalib extends Cross[PanamaLib](v.scalaCrossVersions)

trait PanamaLib extends common.PanamaLibModule with CrossModuleBase with ChiselPublishModule with ScalafmtModule {
  def circtPanamaBindingModule = circtpanamabinding
}

object panamaom extends Cross[PanamaOM](v.scalaCrossVersions)

trait PanamaOM extends common.PanamaOMModule with CrossModuleBase with ChiselPublishModule with ScalafmtModule {
  def panamaLibModule = panamalib(crossScalaVersion)
}

object panamaconverter extends Cross[PanamaConverter](v.scalaCrossVersions)

trait PanamaConverter
    extends common.PanamaConverterModule
    with CrossModuleBase
    with ChiselPublishModule
    with ScalafmtModule {
  def panamaOMModule = panamaom(crossScalaVersion)

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object litutility extends Cross[LitUtility](v.scalaCrossVersions)

trait LitUtility extends tests.LitUtilityModule with CrossModuleBase with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "lit" / "utility"
  def panamaConverterModule = panamaconverter(crossScalaVersion)
  def panamaOMModule = panamaom(crossScalaVersion)
}

object lit extends Cross[Lit](v.scalaCrossVersions)

trait Lit extends tests.LitModule with Cross.Module[String] {
  def scalaVersion: T[String] = crossValue
  def runClasspath: T[Seq[os.Path]] = T(litutility(crossValue).runClasspath().map(_.path))
  def pluginJars:   T[Seq[os.Path]] = T(Seq(litutility(crossValue).panamaConverterModule.pluginModule.jar().path))
  def javaLibraryPath: T[Seq[os.Path]] = T(
    litutility(crossValue).panamaConverterModule.circtPanamaBindingModule.libraryPaths().map(_.path)
  )
  def javaHome:     T[os.Path] = T(os.Path(sys.props("java.home")))
  def chiselLitDir: T[os.Path] = T(millSourcePath)
  def litConfigIn:  T[PathRef] = T.source(millSourcePath / "tests" / "lit.site.cfg.py.in")
}

object benchmark extends ScalaModule with JmhModule with ScalafmtModule {
  def scalaVersion = v.scalaVersion
  def jmhCoreVersion = v.jmhVersion

  override def moduleDeps = Seq(chisel(v.scalaVersion))
}
