import mill._
import mill.scalalib._
import mill.scalalib.TestModule._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import coursier.maven.MavenRepository
import mill.scalalib.api.ZincWorkerUtil.matchingVersions
import $file.common

object v {
  val pluginScalaCrossVersions = Seq(
    "2.13.11",
    "2.13.12"
  )
  val scalaCrossVersions = Seq(
    "2.13.12"
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
        case _ => None
      }.get
    } catch {
      case e: java.io.IOException => None
    }
  }

  def buildVersion = T("build-from-source")

  private def generateBuildInfo = T {
    val outputFile = T.dest / "chisel3" / "BuildInfo.scala"
    val firtoolVersionString = firtoolVersion().map("Some(" + _ + ")").getOrElse("None")
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

object circtpanamabinder extends Cross[CIRCTPanamaBinder](v.scalaCrossVersions)

trait CIRCTPanamaBinder
  extends common.CIRCTPanamaBinderModule
    with ChiselPublishModule
    with CrossSbtModule
    with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "binder"

  def header = T(PathRef(millSourcePath / "jextract-headers.h"))

  def circtInstallPath = T(os.Path(T.ctx.env.get("CIRCT_INSTALL_PATH").getOrElse("/usr/local")))

  def includePaths = T(Seq(PathRef(circtInstallPath() / "include")))

  def libraryPaths = T(Seq(PathRef(circtInstallPath() / "lib")))

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object bindertest extends Cross[CIRCTPanamaBinderModuleTest](v.scalaCrossVersions)

trait CIRCTPanamaBinderModuleTest
  extends common.CIRCTPanamaBinderModuleTestModule
    with CrossModuleBase
    with ScalafmtModule {
  override def millSourcePath = circtpanamabinder(crossScalaVersion).millSourcePath

  def circtPanamaBinderModule = circtpanamabinder(crossScalaVersion)

  def scalatestIvy = v.scalatest

  def scalacheckIvy = v.scalacheck

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "src" / "test"))
  }
}
