// Build script for mill 0.6.0
import mill._
import mill.scalalib._
import mill.scalalib.publish._
import coursier.maven.MavenRepository
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:$MILL_VERSION`
import mill.contrib.buildinfo.BuildInfo

object chisel3 extends mill.Cross[chisel3CrossModule]("2.11.12", "2.12.11") 

// The following stanza is searched for and used when preparing releases.
// Please retain it.
// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map("firrtl" -> "1.4-SNAPSHOT")

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}

// Since chisel contains submodule core and macros, a CommonModule is needed
trait CommonModule extends ScalaModule with SbtModule with PublishModule {
  def firrtlModule: Option[PublishModule]

  def publishVersion = "3.4-SNAPSHOT"

  // 2.12.11 -> Array("2", "12", "10") -> "12" -> 12
  protected def majorVersion = crossVersion.split('.')(1).toInt

  def crossVersion: String

  def scalaVersion = crossVersion

  def repositories() = super.repositories ++ Seq(
    MavenRepository("https://oss.sonatype.org/content/repositories/snapshots"),
    MavenRepository("https://oss.sonatype.org/content/repositories/releases")
  )

  private def scalacCrossOptions = majorVersion match {
    case i if i < 12 => Seq()
    case _ => Seq("-Xsource:2.11")
  }
  
  def ivyDeps = if(firrtlModule.isEmpty) Agg(
    getVersion("firrtl"),
  ) else Agg.empty[Dep]

  def moduleDeps = Seq() ++ firrtlModule

  private def javacCrossOptions = majorVersion match {
    case i if i < 12 => Seq("-source", "1.7", "-target", "1.7")
    case _ => Seq("-source", "1.8", "-target", "1.8")
  }

  override def scalacOptions = super.scalacOptions() ++ Agg(
    "-deprecation",
    "-feature"
  ) ++ scalacCrossOptions
  
  override def javacOptions = super.javacOptions() ++ javacCrossOptions

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

class chisel3CrossModule(crossVersionValue: String) extends CommonModule with PublishModule with BuildInfo { m =>
  // different scala version shares same sources
  // mill use foo/2.11.12 foo/2.12.11 as millSourcePath by default
  override def millSourcePath = super.millSourcePath / os.up / os.up

  def crossVersion = crossVersionValue

  def mainClass = Some("chisel3.stage.ChiselMain")

  def firrtlModule: Option[PublishModule] = None

  override def moduleDeps = super.moduleDeps ++ Seq(macros, core) ++ firrtlModule
  
  object test extends Tests {
    private def ivyCrossDeps = majorVersion match {
      case i if i < 12 => Agg(ivy"junit:junit:4.13")
      case _ => Agg()
    }

    def ivyDeps = Agg(
      ivy"org.scalatest::scalatest:3.1.2",
      ivy"org.scalatestplus::scalacheck-1-14:3.1.1.1",
      ivy"com.github.scopt::scopt:3.7.1"
    ) ++ ivyCrossDeps

    def testFrameworks = Seq("org.scalatest.tools.Framework")

    // a sbt-like testOnly command.
    // for example, mill -i "chisel3[2.12.11].test.testOnly" "chiselTests.BitwiseOpsSpec" 
    def testOnly(args: String*) = T.command {
      super.runMain("org.scalatest.run", args: _*)
    }
  }

  override def buildInfoPackageName = Some("chisel3")

  override def buildInfoMembers: T[Map[String, String]] = T {
    Map(
      "buildInfoPackage" -> artifactName(),
      "version" -> publishVersion(),
      "scalaVersion" -> scalaVersion()
    )
  }

  override def generatedSources = T {
    Seq(generatedBuildInfo()._2)
  }

  object macros extends CommonModule {
    def firrtlModule = m.firrtlModule

    def crossVersion = crossVersionValue
  }

  object core extends CommonModule { 
    def firrtlModule = m.firrtlModule

    def crossVersion = crossVersionValue

    def moduleDeps = super.moduleDeps ++ Seq(macros) ++ firrtlModule

    def scalacOptions = super.scalacOptions() ++ Seq(
      "-deprecation",
      "-explaintypes",
      "-feature",
      "-language:reflectiveCalls",
      "-unchecked",
      "-Xcheckinit",
      "-Xlint:infer-any"
    )
  }
  // make mill publish sbt compatible package
  def artifactName = "chisel3"
}
