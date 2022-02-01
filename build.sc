import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.scalalib.scalafmt._
import coursier.maven.MavenRepository
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:$MILL_VERSION`
import mill.contrib.buildinfo.BuildInfo

object chisel3 extends mill.Cross[chisel3CrossModule]("2.13.6", "2.12.14")

object v {
  val firrtl = ivy"edu.berkeley.cs::firrtl:1.6-SNAPSHOT"
  val treadle = ivy"edu.berkeley.cs::treadle:1.6-SNAPSHOT"
  val chiseltest = ivy"edu.berkeley.cs::chiseltest:0.6-SNAPSHOT"
  val scalatest = ivy"org.scalatest::scalatest:3.2.11"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-14:3.2.2.0"
  val osLib = ivy"com.lihaoyi::os-lib:0.8.0"
  val macroParadise = ivy"org.scalamacros:::paradise:2.1.1"
}

// Since chisel contains submodule core and macros, a CommonModule is needed
trait CommonModule extends CrossSbtModule with PublishModule with ScalafmtModule {
  def firrtlModule: Option[PublishModule] = None

  def firrtlIvyDeps = if (firrtlModule.isEmpty) Agg(
    v.firrtl
  ) else Agg.empty[Dep]

  def treadleModule: Option[PublishModule] = None

  def treadleIvyDeps = if (treadleModule.isEmpty) Agg(
    v.chiseltest
  ) else Agg.empty[Dep]

  def chiseltestModule: Option[PublishModule] = None

  def chiseltestIvyDeps = if (chiseltestModule.isEmpty) Agg(
    v.chiseltest
  ) else Agg.empty[Dep]


  override def moduleDeps = super.moduleDeps ++ firrtlModule

  override def ivyDeps = super.ivyDeps() ++ Agg(
    v.osLib,
  ) ++ firrtlIvyDeps

  def publishVersion = "3.6-SNAPSHOT"

  // 2.12.10 -> Array("2", "12", "10") -> "12" -> 12
  protected def majorVersion = crossScalaVersion.split('.')(1).toInt

  override def repositories = super.repositories ++ Seq(
    MavenRepository("https://oss.sonatype.org/content/repositories/snapshots"),
    MavenRepository("https://oss.sonatype.org/content/repositories/releases")
  )

  override def scalacOptions = T {
    super.scalacOptions() ++ Agg(
      "-deprecation",
      "-feature"
    ) ++ (if (majorVersion == 13) Agg("-Ymacro-annotations") else Agg.empty[String])
  }


  override def compileIvyDeps = if(majorVersion == 13) super.compileIvyDeps else Agg(v.macroParadise)

  override def scalacPluginIvyDeps = if(majorVersion == 13) super.compileIvyDeps else Agg(v.macroParadise)

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

class chisel3CrossModule(val crossScalaVersion: String) extends CommonModule with BuildInfo {
  m =>
  /** Default behavior assumes `build.sc` in the upper path of `src`.
    * This override makes `src` folder stay with `build.sc` in the same directory,
    * If chisel3 is used as a sub-project, [[millSourcePath]] should be overridden to the folder where `src` located.
    */
  override def millSourcePath = super.millSourcePath / os.up

  override def mainClass = T {
    Some("chisel3.stage.ChiselMain")
  }

  override def moduleDeps = super.moduleDeps ++ Seq(macros, core)

  override def scalacPluginClasspath = T {
    super.scalacPluginClasspath() ++ Agg(
      plugin.jar()
    )
  }

  object test extends Tests with TestModule.ScalaTest {
    override def scalacPluginClasspath = m.scalacPluginClasspath

    override def ivyDeps = m.ivyDeps() ++ Agg(
      v.scalatest,
      v.scalacheck,
    ) ++ m.treadleIvyDeps

    override def moduleDeps = super.moduleDeps ++ treadleModule
  }

  object `integration-tests` extends Tests with TestModule.ScalaTest {
    override def scalacPluginClasspath = m.scalacPluginClasspath

    override def sources = T.sources(millSourcePath / "integration-tests" / "src" / "test" / "scala")
    override def ivyDeps = m.ivyDeps() ++ Agg(
      v.scalatest,
      v.scalacheck,
    ) ++ m.treadleIvyDeps ++ m.chiseltestIvyDeps

    override def moduleDeps = super.moduleDeps ++ treadleModule ++ chiseltestModule
  }

  override def buildInfoPackageName = Some("chisel3")

  override def buildInfoMembers = T {
    Map(
      "buildInfoPackage" -> artifactName(),
      "version" -> publishVersion(),
      "scalaVersion" -> scalaVersion()
    )
  }

  object macros extends CommonModule {
    /** millOuterCtx.segment.pathSegments didn't detect error here. */
    override def millSourcePath = m.millSourcePath / "macros"

    override def crossScalaVersion = m.crossScalaVersion

    override def firrtlModule = m.firrtlModule
  }

  object core extends CommonModule {
    /** millOuterCtx.segment.pathSegments didn't detect error here. */
    override def millSourcePath = m.millSourcePath / "core"

    override def crossScalaVersion = m.crossScalaVersion

    override def moduleDeps = super.moduleDeps ++ Seq(macros)

    override def firrtlModule = m.firrtlModule

    def scalacOptions = T {
      super.scalacOptions() ++ Seq(
        "-deprecation",
        "-explaintypes",
        "-feature",
        "-language:reflectiveCalls",
        "-unchecked",
        "-Xcheckinit",
        "-Xlint:infer-any"
      )
    }

    override def generatedSources = T {
      Seq(generatedBuildInfo()._2)
    }
  }

  object plugin extends CommonModule {
    /** millOuterCtx.segment.pathSegments didn't detect error here. */
    override def millSourcePath = m.millSourcePath / "plugin"

    override def crossScalaVersion = m.crossScalaVersion

    override def firrtlModule = m.firrtlModule

    override def ivyDeps = Agg(
      ivy"${scalaOrganization()}:scala-library:$crossScalaVersion",
    ) ++ (if (majorVersion == 13) Agg(ivy"${scalaOrganization()}:scala-compiler:$crossScalaVersion") else Agg.empty[Dep])

    def scalacOptions = T {
      Seq(
        "-Xfatal-warnings"
      )
    }

    override def artifactName = "chisel3-plugin"
  }

  // make mill publish sbt compatible package
  override def artifactName = "chisel3"
}
