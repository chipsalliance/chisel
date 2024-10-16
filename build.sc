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
  val firtoolResolver = ivy"org.chipsalliance::firtool-resolver:1.3.0"
  val scalatest = ivy"org.scalatest::scalatest:3.2.14"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-15:3.2.11.0"
  val json4s = ivy"org.json4s::json4s-native:4.0.6"
  val dataclass = ivy"io.github.alexarchambault::data-class:0.2.5"
  val commonText = ivy"org.apache.commons:commons-text:1.10.0"
  val scopt = ivy"com.github.scopt::scopt:3.7.1"

  def scalaReflect(scalaVersion: String) = ivy"org.scala-lang:scala-reflect:$scalaVersion"

  def scalaCompiler(scalaVersion: String) = ivy"org.scala-lang:scala-compiler:$scalaVersion"

  def scalaLibrary(scalaVersion: String) = ivy"org.scala-lang:scala-library:$scalaVersion"
<<<<<<< HEAD
=======

  // 21, 1-2, {linux-x64, macos-x64, windows-x64}
  // 22, 1-2, {linux-x64, macos-aarch64, macos-x64, windows-x64}
  def jextract(jdkVersion: Int, jextractVersion: String, os: String, platform: String) =
    s"https://download.java.net/java/early_access/jextract/21/1/openjdk-${jdkVersion}-jextract+${jextractVersion}_${os}-${platform}_bin.tar.gz"

  def circt(version: String, os: String, platform: String) =
    s"https://github.com/llvm/circt/releases/download/firtool-${version}/circt-full-shared-${os}-${platform}.tar.gz"

  val warnConf = Seq(
    "msg=APIs in chisel3.internal:s",
    "msg=Importing from firrtl:s",
    "msg=migration to the MLIR:s",
    "msg=method hasDefiniteSize in trait IterableOnceOps is deprecated:s", // replacement `knownSize` is not in 2.12
    "msg=object JavaConverters in package collection is deprecated:s",
    "msg=undefined in comment for method cf in class PrintableHelper:s",
    // This is deprecated for external users but not internal use
    "cat=deprecation&origin=firrtl\\.options\\.internal\\.WriteableCircuitAnnotation:s",
    "cat=deprecation&origin=chisel3\\.util\\.experimental\\.BoringUtils.*:s",
    "cat=deprecation&origin=chisel3\\.experimental\\.IntrinsicModule:s",
    "cat=deprecation&origin=chisel3\\.ltl.*:s"
  )

  // ScalacOptions
  val commonOptions = Seq(
    "-deprecation",
    "-feature",
    "-unchecked",
    "-Werror",
    "-Ymacro-annotations",
    "-explaintypes",
    "-Xcheckinit",
    "-Xlint:infer-any",
    "-Xlint:missing-interpolator",
    "-language:reflectiveCalls",
    s"-Wconf:${warnConf.mkString(",")}"
  )
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

trait ChiselPublishModule extends CiReleaseModule {
  // Publish information
  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "org.chipsalliance",
    url = "https://www.chisel-lang.org",
    licenses = Seq(License.`Apache-2.0`),
    versionControl = VersionControl.github("chipsalliance", "chisel"),
    developers = Seq(
      Developer("jackkoenig", "Jack Koenig", "https://github.com/jackkoenig"),
      Developer("azidar", "Adam Izraelevitz", "https://github.com/azidar"),
      Developer("seldridge", "Schuyler Eldridge", "https://github.com/seldridge")
    )
  )

  override def sonatypeHost = Some(SonatypeHost.s01)

  override def publishVersion = VcsVersion
    .vcsState()
    .format(
      countSep = "+",
      revHashDigits = 8,
      untaggedSuffix = "-SNAPSHOT"
    )

>>>>>>> ca773c08a (Fix missing string interpolators, add -Xlint:missing-interpolator (#4471))
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

  def firtoolResolverModuleIvy = v.firtoolResolver

  def firtoolVersion = T {
    val contents = os.read(millSourcePath / os.up / "etc" / "circt.json")
    val read = upickle.default.read[Map[String, String]](contents)
    read("version").stripPrefix("firtool-")
  }

  def buildVersion = T("build-from-source")

  private def generateBuildInfo = T {
    val outputFile = T.dest / "chisel3" / "BuildInfo.scala"
    val firtoolVersionString = "Some(\"" + firtoolVersion() + "\")"
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

  def circtInstallPath = T.input(os.Path(T.ctx.env.get("CIRCT_INSTALL_PATH").getOrElse("/usr/local")))

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
