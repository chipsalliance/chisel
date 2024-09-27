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

object v {
  val pluginScalaCrossVersions = Seq(
    "2.13.11",
    "2.13.12",
    "2.13.13",
    "2.13.14",
    "2.13.15"
  )
  val scalaCrossVersions = Seq(
    "2.13.15"
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

  // 21, 1-2, {linux-x64, macos-x64, windows-x64}
  // 22, 1-2, {linux-x64, macos-aarch64, macos-x64, windows-x64}
  def jextract(jdkVersion: Int, jextractVersion: String, os: String, platform: String) =
    s"https://download.java.net/java/early_access/jextract/21/1/openjdk-${jdkVersion}-jextract+${jextractVersion}_${os}-${platform}_bin.tar.gz"

  def circt(version: String, os: String, platform: String) =
    s"https://github.com/llvm/circt/releases/download/firtool-${version}/circt-full-shared-${os}-${platform}.tar.gz"
}

object utils extends Module {

  val architecture = System.getProperty("os.arch")
  val operationSystem = System.getProperty("os.name")

  val mac = operationSystem.toLowerCase.startsWith("mac")
  val linux = operationSystem.toLowerCase.startsWith("linux")
  val windows = operationSystem.toLowerCase.startsWith("win")
  val amd64 = architecture.matches("^(x8664|amd64|ia32e|em64t|x64|x86_64)$")
  val aarch64 = architecture.equals("aarch64") | architecture.startsWith("armv8")

  val firtoolVersion = {
    val j = _root_.upickle.default.read[Map[String, String]](os.read(millSourcePath / os.up / "etc" / "circt.json"))
    j("version").stripPrefix("firtool-")
  }

  // use T.persistent to avoid download repeatedly
  def circtInstallDir: T[os.Path] = T.persistent {
    T.ctx().env.get("CIRCT_INSTALL_PATH") match {
      case Some(dir) => os.Path(dir)
      case None =>
        T.ctx().log.info("Use CIRCT_INSTALL_PATH to vendor circt")
        val tarPath = T.dest / "circt.tar.gz"
        if (!os.exists(tarPath)) {
          val url = v.circt(
            firtoolVersion,
            if (linux) "linux" else if (mac) "macos" else throw new Exception("unsupported os"),
            // circt does not yet publish for macos-aarch64, use x64 for now
            if (amd64 || mac) "x64" else throw new Exception("unsupported arch")
          )
          T.ctx().log.info(s"Downloading circt from ${url}")
          mill.util.Util.download(url, os.rel / "circt.tar.gz")
          T.ctx().log.info(s"Download Successfully")
        }
        os.proc("tar", "xvf", tarPath, "--strip-components=1").call(T.dest)
        T.dest
    }
  }

  // use T.persistent to avoid download repeatedly
  def jextractInstallDir: T[os.Path] = T.persistent {
    T.ctx().env.get("JEXTRACT_INSTALL_PATH") match {
      case Some(dir) => os.Path(dir)
      case None =>
        T.ctx().log.info("Use JEXTRACT_INSTALL_PATH to vendor jextract")
        val tarPath = T.dest / "jextract.tar.gz"
        if (!os.exists(tarPath)) {
          val url = v.jextract(
            21,
            "1-2",
            if (linux) "linux" else if (mac) "macos" else throw new Exception("unsupported os"),
            // There is no macos-aarch64 for jextract 21, use x64 for now
            if (amd64 || mac) "x64" else if (aarch64) "aarch64" else throw new Exception("unsupported arch")
          )
          T.ctx().log.info(s"Downloading jextract from ${url}")
          mill.util.Util.download(url, os.rel / "jextract.tar.gz")
          T.ctx().log.info(s"Download Successfully")
        }
        os.proc("tar", "xvf", tarPath, "--strip-components=1").call(T.dest)
        T.dest
    }
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

  object test extends SbtModuleTests with TestModule.ScalaTest with ScalafmtModule {
    def ivyDeps = Agg(v.scalatest, v.scalacheck)
  }
}

object svsim extends Cross[Svsim](v.scalaCrossVersions)

trait Svsim extends common.SvsimModule with ChiselPublishModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "svsim"

  object test extends SbtModuleTests with TestModule.ScalaTest with ScalafmtModule {
    def ivyDeps = Agg(v.scalatest, v.scalacheck)
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
    val firtoolVersionString = "Some(\"" + utils.firtoolVersion + "\")"
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

  object test extends SbtModuleTests with TestModule.ScalaTest with ScalafmtModule {
    def ivyDeps = Agg(v.scalatest, v.scalacheck)
  }
}

object integrationTests extends Cross[IntegrationTests](v.scalaCrossVersions)

trait IntegrationTests extends CrossSbtModule with ScalafmtModule with common.HasChiselPlugin {

  def pluginModule = plugin()

  def millSourcePath = os.pwd / "integration-tests"

  object test extends SbtModuleTests with TestModule.ScalaTest with ScalafmtModule {
    override def moduleDeps = super.moduleDeps :+ chisel().test
    def ivyDeps = Agg(v.scalatest, v.scalacheck)
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

object circtpanamabinding extends CIRCTPanamaBinding

trait CIRCTPanamaBinding extends common.CIRCTPanamaBindingModule with ChiselPublishModule {

  def header = T(PathRef(millSourcePath / "jextract-headers.h"))

  def circtInstallPath = T(utils.circtInstallDir())

  def jextractBinary = T(utils.jextractInstallDir() / "bin" / "jextract")

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
