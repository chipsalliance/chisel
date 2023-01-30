import mill._
import mill.scalalib._
import mill.scalalib.TestModule._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import coursier.maven.MavenRepository
import $file.common

// The following stanza is searched for and used when preparing releases.
// Please retain it.
// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map(
  "firrtl" -> "1.6-SNAPSHOT",
  "treadle" -> "1.6-SNAPSHOT"
)

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}
// Do not remove the above logic, it is needed by the release automation

object v {
  val pluginScalaCrossVersions = Seq(
    // scalamacros paradise version used is not published for 2.12.0 and 2.12.1
    "2.12.2",
    "2.12.3",
    // 2.12.4 is broken in newer versions of Zinc: https://github.com/sbt/sbt/issues/6838
    "2.12.5",
    "2.12.6",
    "2.12.7",
    "2.12.8",
    "2.12.9",
    "2.12.10",
    "2.12.11",
    "2.12.12",
    "2.12.13",
    "2.12.14",
    "2.12.15",
    "2.12.16",
    "2.12.17",
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
    "2.13.10"
  )
  val scalaCrossVersions = Seq(
    "2.12.17",
    "2.13.10"
  )
  val firtoolCrossVersions = Seq("1.27.0")
  val firrtl = getVersion("firrtl")
  val osLib = ivy"com.lihaoyi::os-lib:0.8.1"
  val upickle = ivy"com.lihaoyi::upickle:2.0.0"
  val macroParadise = ivy"org.scalamacros:::paradise:2.1.1"
  val treadle = getVersion("treadle")
  val chiseltest = ivy"edu.berkeley.cs::chiseltest:0.6-SNAPSHOT"
  val scalatest = ivy"org.scalatest::scalatest:3.2.15"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-14:3.2.2.0"

  def scalaReflect(scalaVersion: String) = ivy"org.scala-lang:scala-reflect:$scalaVersion"

  def scalaCompiler(scalaVersion: String) = ivy"org.scala-lang:scala-compiler:$scalaVersion"

  def scalaLibrary(scalaVersion: String) = ivy"org.scala-lang:scala-library:$scalaVersion"
}
private def majorScalaVersion(scalaVersion: String) = scalaVersion.split('.')(1).toInt

object macros extends mill.Cross[macros](v.scalaCrossVersions: _*)

class macros(val crossScalaVersion: String)
  extends common.MacrosModule
    with CrossSbtModule
    with ScalafmtModule {
  def scalaReflectIvy= v.scalaReflect(crossScalaVersion)

  def macroParadiseIvy: Option[Dep] = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None
}

object core extends mill.Cross[core](v.scalaCrossVersions: _*)

class core(val crossScalaVersion: String)
  extends common.CoreModule
    with CrossSbtModule
    with ScalafmtModule
    with SonatypeSnapshotModule {
  def firrtlIvyDeps = Some(v.firrtl)

  def macrosModule = macros(crossScalaVersion)

  def macroParadiseIvy: Option[Dep] = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None

  def osLibModuleIvy = v.osLib

  def upickleModuleIvy = v.upickle
}

object plugin extends mill.Cross[plugin](v.pluginScalaCrossVersions: _*)

class plugin(val crossScalaVersion: String)
  extends common.PluginModule
    with CrossSbtModule
    with ScalafmtModule {
  def scalaLibraryIvy = v.scalaLibrary(crossScalaVersion)

  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)

  def scalaCompilerIvy: Dep = v.scalaCompiler(crossScalaVersion)
}

object chisel3 extends mill.Cross[chisel3](v.scalaCrossVersions: _*)

class chisel3(val crossScalaVersion: String)
  extends common.ChiselModule
    with CrossSbtModule
    with ScalafmtModule
    with SonatypeSnapshotModule {
  override def millSourcePath = super.millSourcePath / os.up

  def macrosModule = macros(crossScalaVersion)

  def coreModule = core(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)

  def macroParadiseIvy = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None

  object test
    extends Tests
      with common.HasChiselPlugin
      with ScalafmtModule
      with ScalaTest {
    def pluginModule = plugin(crossScalaVersion)

    override def ivyDeps = {
      super.ivyDeps() ++
        Seq(
          v.chiseltest,
          v.scalacheck,
          v.scalatest
        )
    }
  }

  object `integration-tests`
    extends Tests
      with common.HasChiselPlugin
      with ScalafmtModule
      with ScalaTest {
    override def sources = T.sources(millSourcePath / "integration-tests" / "src" / "test" / "scala")

    def pluginModule = plugin(crossScalaVersion)

    override def moduleDeps: Seq[JavaModule] = super.moduleDeps ++ Seq(stdlib(crossScalaVersion))

    override def ivyDeps = {
      super.ivyDeps() ++
        Seq(
          v.chiseltest,
          v.scalacheck,
          v.scalatest
        )
    }
  }
}

object stdlib extends mill.Cross[stdlib](v.scalaCrossVersions: _*)

class stdlib(val crossScalaVersion: String)
  extends common.StdLibModule
    with CrossSbtModule
    with ScalafmtModule
    with SonatypeSnapshotModule {
  def chiselModule = chisel3(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}

object circt extends mill.Cross[circt](v.firtoolCrossVersions: _*)

class circt(firtoolCrossVersion: String)
  extends common.CIRCTModule { cm =>
  def circtSourcePath: T[PathRef] = T {
    val circtPath = T.dest / s"circt"
    os.proc("git", "clone", "https://github.com/llvm/circt", "--depth", "1", "--branch", s"firtool-${firtoolCrossVersion}", circtPath).call(T.dest)
    PathRef(circtPath)
  }

  def llvmSourcePath: T[PathRef] = T {
    val llvmPath = circtSourcePath().path / s"llvm"
    os.proc("git", "submodule", "init", "llvm").call(circtSourcePath().path)
    os.proc("git", "submodule", "update", "--depth", 1).call(circtSourcePath().path)
    PathRef(circtSourcePath().path / s"llvm")
  }
}

object `circt-panama` extends mill.Cross[`circt-panama`](v.firtoolCrossVersions: _*)

class `circt-panama`(firtoolCrossVersion: String)
  extends common.CIRCTPanamaModule {
  def circtModule = circt(firtoolCrossVersion)
}

object `chisel-circt-panama` extends mill.Cross[`chisel-circt-panama`]((for {
  scalaCrossVersion <- v.scalaCrossVersions
  firtoolCrossVersion <- v.firtoolCrossVersions
} yield (scalaCrossVersion, firtoolCrossVersion)): _*)

class `chisel-circt-panama`(val crossScalaVersion: String, firtoolCrossVersion: String)
  extends common.ChiselCIRCTPanamaModule
    with CrossSbtModule
    with ScalafmtModule
    with SonatypeSnapshotModule {
  def scalaVersion = crossScalaVersion
  def coreModule = core(crossScalaVersion)
  def circtPanamaModule = `circt-panama`(firtoolCrossVersion)
  def circtInstallDirectory = `circt-panama`(firtoolCrossVersion).circtInstallDirectory
}


trait SonatypeSnapshotModule extends CoursierModule {
  override def repositoriesTask = T.task {
    super.repositoriesTask() ++ Seq(
      MavenRepository("https://oss.sonatype.org/content/repositories/snapshots"),
      MavenRepository("https://oss.sonatype.org/content/repositories/releases")
    )
  }
}

trait ChiselPublishModule extends PublishModule {
  def publishVersion = "3.6-SNAPSHOT"

  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "edu.berkeley.cs",
    url = "https://www.chisel-lang.org",
    licenses = Seq(License.`Apache-2.0`),
    versionControl = VersionControl.github("freechipsproject", "chisel3"),
    developers = Seq(
      Developer("jackbackrack", "Jonathan Bachrach", "https://eecs.berkeley.edu/~jrb/")
    )
  )
}
