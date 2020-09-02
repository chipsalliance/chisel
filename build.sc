import mill._
import mill.scalalib._
import mill.scalalib.publish._
import coursier.maven.MavenRepository
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:$MILL_VERSION`
import mill.contrib.buildinfo.BuildInfo

object chisel3 extends mill.Cross[chisel3CrossModule]("2.11.12", "2.12.12")

// The following stanza is searched for and used when preparing releases.
// Please retain it.
// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map(
  "firrtl" -> "1.4-SNAPSHOT"
)

val testDefaultVersions = Map(
  "treadle" -> "1.3-SNAPSHOT"
)

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}

def getTestVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", testDefaultVersions(dep))
  ivy"$org::$dep:$version"
}

// Since chisel contains submodule core and macros, a CommonModule is needed
trait CommonModule extends CrossSbtModule with PublishModule {
  def firrtlModule: Option[PublishModule] = None

  def firrtlIvyDeps = if (firrtlModule.isEmpty) Agg(
    getVersion("firrtl")
  ) else Agg.empty[Dep]

  def treadleModule: Option[PublishModule] = None

  def treadleIvyDeps = if (treadleModule.isEmpty) Agg(
    getTestVersion("treadle")
  ) else Agg.empty[Dep]

  override def moduleDeps = super.moduleDeps ++ firrtlModule

  override def ivyDeps = super.ivyDeps() ++ firrtlIvyDeps

  def publishVersion = "3.4-SNAPSHOT"

  // 2.12.10 -> Array("2", "12", "10") -> "12" -> 12
  protected def majorVersion = crossScalaVersion.split('.')(1).toInt

  override def repositories = super.repositories ++ Seq(
    MavenRepository("https://oss.sonatype.org/content/repositories/snapshots"),
    MavenRepository("https://oss.sonatype.org/content/repositories/releases")
  )

  private def scalacCrossOptions = majorVersion match {
    case i if i < 12 => Seq()
    case _ => Seq("-Xsource:2.11")
  }

  private def javacCrossOptions = majorVersion match {
    case i if i < 12 => Seq("-source", "1.7", "-target", "1.7")
    case _ => Seq("-source", "1.8", "-target", "1.8")
  }

  override def scalacOptions = T {
    super.scalacOptions() ++ Agg(
      "-deprecation",
      "-feature"
    ) ++ scalacCrossOptions
  }

  override def javacOptions = T {
    super.javacOptions() ++ javacCrossOptions
  }

  private val macroParadise = ivy"org.scalamacros:::paradise:2.1.1"

  override def compileIvyDeps = Agg(macroParadise)

  override def scalacPluginIvyDeps = Agg(macroParadise)

  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "edu.berkeley.cs",
    url = "https://www.chisel-lang.org",
    licenses = Seq(License.`BSD-3-Clause`),
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

  object test extends Tests {
    override def scalacPluginClasspath: T[Loose.Agg[PathRef]] = m.scalacPluginClasspath

    private def ivyCrossDeps = majorVersion match {
      case i if i < 12 => Agg(ivy"junit:junit:4.13")
      case _ => Agg()
    }

    override def ivyDeps = m.ivyDeps() ++ Agg(
      ivy"org.scalatest::scalatest:3.1.2",
      ivy"org.scalatestplus::scalacheck-1-14:3.1.1.1",
      ivy"com.github.scopt::scopt:3.7.1"
    ) ++ ivyCrossDeps ++ m.treadleIvyDeps

    override def moduleDeps = super.moduleDeps ++ treadleModule

    def testFrameworks = T {
      Seq("org.scalatest.tools.Framework")
    }

    // a sbt-like testOnly command.
    // for example, mill -i "chisel3[2.12.12].test.testOnly" "chiselTests.BitwiseOpsSpec"
    def testOnly(args: String*) = T.command {
      super.runMain("org.scalatest.run", args: _*)
    }
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
      ivy"${scalaOrganization()}:scala-library:$crossScalaVersion"
    )

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
