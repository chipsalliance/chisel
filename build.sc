import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import mill.api.Result
import mill.define.Cross
import mill.scalalib.api.ZincWorkerUtil.matchingVersions
import mill.util.Jvm.createJar
import $ivy.`com.lihaoyi::mill-contrib-jmh:`
import mill.contrib.jmh.JmhModule
import $ivy.`io.chris-kipp::mill-ci-release::0.1.10`
import io.kipp.mill.ci.release.{CiReleaseModule, SonatypeHost}
import de.tobiasroeser.mill.vcs.version.VcsVersion // pulled in by mill-ci-release
import $file.common
import $file.tests

object v {

  val javaVersion = {
    val rawVersion = sys.props("java.specification.version")
    // Older versions of Java started with 1., e.g. 1.8 == 8
    rawVersion.stripPrefix("1.").toInt
  }

  // Java 21 only works with 2.13.11+, but Project Panama uses Java 21
  // Only publish plugin for 2.13.11+ when using Java > 11, but still
  // publish all versions when Java version <= 11.
  val pluginScalaCrossVersions = {
    val latest213 = 15
    val java21Min213 = 11
    val minVersion = if (javaVersion > 11) java21Min213 else 0
    val versions = minVersion to latest213
    versions.map(v => s"2.13.$v").toSeq
  }
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

}

object firrtl extends Cross[Firrtl](v.scalaCrossVersions)

trait Firrtl extends common.FirrtlModule with CrossSbtModule with ScalafmtModule {
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

trait Svsim extends common.SvsimModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "svsim"

  object test extends SbtModuleTests with TestModule.ScalaTest with ScalafmtModule {
    def ivyDeps = Agg(v.scalatest, v.scalacheck)
  }
}

object macros extends Cross[Macros](v.scalaCrossVersions)

trait Macros extends common.MacrosModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "macros"

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)
}

object core extends Cross[Core](v.scalaCrossVersions)

trait Core extends common.CoreModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "core"

  def firrtlModule = firrtl(crossScalaVersion)

  def macrosModule = macros(crossScalaVersion)

  def osLibModuleIvy = v.osLib

  def upickleModuleIvy = v.upickle

  def firtoolResolverModuleIvy = v.firtoolResolver

  // Similar to the publish version, but no dirty indicators because otherwise
  // this file will change any time any file is changed.
  def publishVersion = T {
    VcsVersion
      .vcsState()
      .format(
        countSep = "+",
        revHashDigits = 8,
        untaggedSuffix = "-SNAPSHOT",
        dirtySep = "",
        dirtyHashDigits = 0
      )
  }

  def buildInfo = T {
    val outputFile = T.dest / "chisel3" / "BuildInfo.scala"
    val firtoolVersionString = "Some(\"" + utils.firtoolVersion + "\")"
    val contents =
      s"""
         |package chisel3
         |case object BuildInfo {
         |  val buildInfoPackage: String = "chisel3"
         |  val version: String = "${publishVersion()}"
         |  val scalaVersion: String = "${scalaVersion()}"
         |  @deprecated("Chisel no longer uses SBT, this field will be removed.", "Chisel 7.0")
         |  val sbtVersion: String = ""
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
    super.generatedSources() :+ buildInfo()
  }
}

object plugin extends Cross[Plugin](v.pluginScalaCrossVersions)

trait Plugin extends common.PluginModule with CrossSbtModule with ScalafmtModule with ChiselPublishModule {
  override def artifactName = "chisel-plugin"

  // The plugin is compiled for every minor Scala version
  override def crossFullScalaVersion = true

  def millSourcePath = super.millSourcePath / os.up / "plugin"

  def scalaLibraryIvy = v.scalaLibrary(crossScalaVersion)

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)

  def scalaCompilerIvy: Dep = v.scalaCompiler(crossScalaVersion)
}

object chisel extends Cross[Chisel](v.scalaCrossVersions)

trait Chisel extends common.ChiselModule with CrossSbtModule with ScalafmtModule {
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

trait Stdlib extends common.StdLibModule with CrossSbtModule with ScalafmtModule {
  def millSourcePath = super.millSourcePath / os.up / "stdlib"

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object circtpanamabinding extends CIRCTPanamaBinding

trait CIRCTPanamaBinding extends common.CIRCTPanamaBindingModule {

  def header = T(PathRef(millSourcePath / "jextract-headers.h"))

  def circtInstallPath = T(utils.circtInstallDir())

  def jextractBinary = T(utils.jextractInstallDir() / "bin" / "jextract")

  def includePaths = T(Seq(PathRef(circtInstallPath() / "include")))

  def libraryPaths = T(Seq(PathRef(circtInstallPath() / "lib")))
}

object panamalib extends Cross[PanamaLib](v.scalaCrossVersions)

trait PanamaLib extends common.PanamaLibModule with CrossModuleBase with ScalafmtModule {
  def circtPanamaBindingModule = circtpanamabinding
}

object panamaom extends Cross[PanamaOM](v.scalaCrossVersions)

trait PanamaOM extends common.PanamaOMModule with CrossModuleBase with ScalafmtModule {
  def panamaLibModule = panamalib(crossScalaVersion)
}

object panamaconverter extends Cross[PanamaConverter](v.scalaCrossVersions)

trait PanamaConverter extends common.PanamaConverterModule with CrossModuleBase with ScalafmtModule {
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

/** Aggregate project for publishing Chisel as a single artifact
  */
object unipublish extends ScalaModule with ChiselPublishModule {

  def scalaVersion = v.scalaVersion

  // This is published as chisel
  override def artifactName = "chisel"

  // Older versions of Scala do not work with newer versions of the JVM
  // This is a hack to ensure we always use Java 8 to publish Chisel with Scala 2.13
  // We could use Java 11 with -release 8
  // Note that this target is used by real publishing but not by publishLocal
  override def publishArtifacts = T {
    // TODO when we publish for Scala 3, only do this check for Scala 2.13
    if (v.javaVersion != 8) {
      throw new Exception(s"Publishing requires Java 8, current JDK is ${v.javaVersion}")
    }
    super.publishArtifacts
  }

  /** Publish both this project and the plugin (for the default Scala version) */
  override def publishLocal(localIvyRepo: String = null) = T.command {
    // TODO consider making this parallel and publishing all cross-versions for plugin
    plugin(v.scalaVersion).publishLocal(localIvyRepo)()
    super.publishLocal(localIvyRepo)()
  }

  // Explicitly not using moduleDeps because that influences so many things
  def components = Seq(firrtl, svsim, macros, core, chisel).map(_(v.scalaVersion))

  /** Aggregated ivy deps to include as dependencies in POM */
  def ivyDeps = T { T.traverse(components)(_.ivyDeps)().flatten }

  /** Aggregated local classpath to include in jar */
  override def localClasspath = T { T.traverse(components)(_.localClasspath)().flatten }

  /** Aggreagted sources from all component modules */
  def aggregatedSources = T { T.traverse(components)(_.allSources)().flatten }

  /** Aggreagted resources from all component modules */
  def aggregatedResources = T { T.traverse(components)(_.resources)().flatten }

  /** Aggreagted compile resources from all component modules */
  def aggregatedCompileResources = T { T.traverse(components)(_.compileResources)().flatten }

  /** Aggregated sourceJar from all component modules
    */
  override def sourceJar: T[PathRef] = T {
    // This is based on the implementation of sourceJar in PublishModule, may need to be kept in sync.
    val allDirs = aggregatedSources() ++ aggregatedResources() ++ aggregatedCompileResources()
    createJar(allDirs.map(_.path).filter(os.exists), manifest())
  }

  // Needed for ScalaDoc
  override def scalacOptions = T {
    Seq("-Ymacro-annotations")
  }

  def scalaDocRootDoc = T.source { T.workspace / "root-doc.txt" }

  def unidocOptions = T {
    scalacOptions() ++ Seq[String](
      "-classpath",
      unidocCompileClasspath().map(_.path).mkString(sys.props("path.separator")),
      "-diagrams",
      "-groups",
      "-skip-packages",
      "chisel3.internal",
      "-diagrams-max-classes",
      "25",
      "-doc-version",
      publishVersion(),
      "-doc-title",
      "chisel",
      "-doc-root-content",
      scalaDocRootDoc().path.toString,
      "-sourcepath",
      T.workspace.toString,
      "-doc-source-url",
      unidocSourceUrl(),
      "-language:implicitConversions",
      "-implicits"
    )
  }

  // Built-in UnidocModule is insufficient so we need to implement it ourselves
  // We could factor this out into a utility
  def unidocSourceUrl: T[String] = T {
    val base = "https://github.com/chipsalliance/chisel/tree"
    val branch = if (publishVersion().endsWith("-SNAPSHOT")) "main" else s"v${publishVersion()}"
    s"$base/$branch/€{FILE_PATH_EXT}#L€{FILE_LINE}"
  }

  def unidocVersion: T[Option[String]] = None

  def unidocCompileClasspath = T {
    Seq(compile().classes) ++ T.traverse(components)(_.compileClasspath)().flatten
  }

  def unidocSourceFiles = T {
    allSourceFiles() ++ T.traverse(components)(_.allSourceFiles)().flatten
  }

  // Based on UnidocModule and docJar in Mill, may need to be kept in sync.
  override def docJar = T {
    T.log.info(s"Building unidoc for ${unidocSourceFiles().length} files")

    val javadocDir = T.dest / "javadoc"
    os.makeDir(javadocDir)

    val fullOptions = unidocOptions() ++
      Seq("-d", javadocDir.toString) ++
      unidocSourceFiles().map(_.path.toString)

    zincWorker()
      .worker()
      .docJar(
        scalaVersion(),
        scalaOrganization(),
        scalaDocClasspath(),
        scalacPluginClasspath(),
        fullOptions
      ) match {
      case true  => Result.Success(createJar(Agg(javadocDir))(T.dest))
      case false => Result.Failure("docJar generation failed")
    }
  }
}
