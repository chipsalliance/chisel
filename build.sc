import ammonite.ops._
import ammonite.ops.ImplicitWd._
import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.eval.{Evaluator, Result}
import mill.define.Task
import mill.modules.Jvm._

import $file.CommonBuild

// Define our own BuildInfo since mill doesn't currently have one.
// It would be nice if we could put this in a separate file, but
//  we encounter problems with `illegal inheritance from sealed trait BuildInfo`
//  when it's in a separate file.
trait BuildInfo extends ScalaModule { outer =>

  def buildInfoPackageName: Option[String] = None

  def buildInfoObjectName: String = "BuildInfo"

  def buildInfoMembers: T[Map[String, String]] = T {
    Map.empty[String, String]
  }

  private def generateBuildInfo(outputPath: Path, members: Map[String, String]) = {
    val outputFile = outputPath / "BuildInfo.scala"
    val internalMembers =
      members
        .map {
          case (name, value) => s"""  val ${name}: String = "${value}""""
        }
        .mkString("\n")
    write(outputFile,
      s"""
         |${buildInfoPackageName.map(p => s"package ${p}").getOrElse("")}
         |case object ${buildInfoObjectName}{
         |$internalMembers
         |  override val toString: String = {
         |    "buildInfoPackage: %s, version: %s, scalaVersion: %s" format (
         |        buildInfoPackage, version, scalaVersion
         |    )
         |  }
         |}
       """.stripMargin)
    outputPath
  }

  override def generatedSources = T {
    super.generatedSources() :+ PathRef(generateBuildInfo(T.ctx().dest, buildInfoMembers()))
  }
}

trait ThisBuildInfo extends BuildInfo with CrossModuleBase{
  override def buildInfoPackageName = Some("chisel3")
  override def buildInfoMembers = T {
    Map[String, String](
      "buildInfoPackage" -> buildInfoPackageName.get,
      "version" -> "3.2-SNAPSHOT",
      "scalaVersion" -> scalaVersion()
    )
  }
}

// An sbt layout with src in the top directory.
trait CrossUnRootedSbtModule extends CrossSbtModule {
  override def millSourcePath = super.millSourcePath / ammonite.ops.up
}

object chiselCompileOptions {
  def scalacOptions = Seq(
    "-deprecation",
    "-explaintypes",
    "-feature",
    "-language:reflectiveCalls",
    "-unchecked",
    "-Xcheckinit",
    "-Xlint:infer-any"
/*    "-Xlint:missing-interpolator" // this causes a:
//[error] .../chisel3/chiselFrontend/src/main/scala/chisel3/core/Aggregate.scala:605:48: recursive value outer needs type
//[error]             val outer = clazz.getDeclaredField("$outer").get(this)
//[error]                                                ^
//[error] one error found
 */
  )
}

// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map("firrtl" -> "1.2-SNAPSHOT")

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}

// Define the common chisel module.
trait CommonChiselModule extends CrossSbtModule {
  override def scalacOptions = chiselCompileOptions.scalacOptions ++ CommonBuild.scalacOptionsVersion(crossScalaVersion)
  override def javacOptions = CommonBuild.javacOptionsVersion(crossScalaVersion)
  val macroPlugins = Agg(ivy"org.scalamacros:::paradise:2.1.0")
  def scalacPluginIvyDeps = macroPlugins
  def compileIvyDeps = macroPlugins
  def chiselDeps = Agg("firrtl").map { d => getVersion(d) }
  override def ivyDeps = chiselDeps
}

trait PublishChiselModule extends CommonChiselModule with PublishModule {
  override def artifactName = "chisel3"
  def publishVersion = "3.2-SNAPSHOT"

  def pomSettings = PomSettings(
    description = artifactName(),
    organization = "edu.berkeley.cs",
    url = "https://chisel.eecs.berkeley.edu",
    licenses = Seq(License.`BSD-3-Clause`),
    versionControl = VersionControl.github("freechipsproject", "chisel3"),
    developers = Seq(
      Developer("jackbackrack",    "Jonathan Bachrach",      "https://eecs.berkeley.edu/~jrb/")
    )
  )
}

// If would be nice if we didn't need to do this, but PublishModule may only be dependent on
//  other PublishModules.
trait UnpublishedChiselModule extends PublishChiselModule

val crossVersions = Seq("2.11.12", "2.12.4")

// Make this available to external tools.
object chisel3 extends Cross[ChiselTopModule](crossVersions: _*) {
  def defaultVersion(ev: Evaluator[Any]) = T.command{
    println(crossVersions.head)
  }

  def compile = T{
    chisel3(crossVersions.head).compile()
  }

  def jar = T{
    chisel3(crossVersions.head).jar()
  }

  def test = T{
    chisel3(crossVersions.head).test.test()
  }

  def publishLocal = T{
    chisel3(crossVersions.head).publishLocal()
  }

  def docJar = T{
    chisel3(crossVersions.head).docJar()
  }
}

object coreMacros extends Cross[CoreMacrosModule](crossVersions: _*) {
}

// Perhaps a better method would be to subsume the ChiselFrontEnd sources,
//  and reduce things to two modules - coreMacros and ChiselTopModule (chisel3).
// We could redefine `sources` in the top module to include `pwd / 'chiselFrontend / 'src`
object chiselFrontend extends Cross[ChiselFrontendModule](crossVersions: _*) {
}

// These submodules follow the `mill` convention - their source is in sub-directory with the module name.
class CoreMacrosModule(val crossScalaVersion: String) extends UnpublishedChiselModule {
}

class ChiselFrontendModule(val crossScalaVersion: String) extends UnpublishedChiselModule {
  override def moduleDeps = Seq(coreMacros(crossScalaVersion))
}

// This submodule is unrooted - its source directory is in the top level directory.
class ChiselTopModule(val crossScalaVersion: String) extends PublishChiselModule with ThisBuildInfo with CrossUnRootedSbtModule {
  override def moduleDeps = Seq(coreMacros(crossScalaVersion), chiselFrontend(crossScalaVersion))

  // In order to preserve our "all-in-one" policy for published jars,
  //  we define allModuleSources() to include transitive sources, and define
  //  allModuleClasspath() to include transitive classes.
  def transitiveSources = T {
    Task.traverse(moduleDeps)(m =>
      T.task{m.allSources()}
    )().flatten
  }

  def allModuleSources = T {
    allSources() ++ transitiveSources()
  }

  def transitiveResources = T {
    Task.traverse(moduleDeps)(m =>
      T.task{m.resources()}
    )().flatten
  }

  def allModuleResources = T {
    resources() ++ transitiveResources()
  }

  // We package all classes in a singe jar.
  def allModuleClasspath = T {
    localClasspath() ++ transitiveLocalClasspath()
  }

  // Define some file filters to exclude unwanted files from created jars.
  type JarFileFilter = (Path, RelPath) => Boolean
  // Exclude any `.DS_Store` files
  val noDS_StoreFiles: JarFileFilter = (p: Path, relPath: RelPath) => {
    relPath.last != ".DS_Store"
  }

  // Exclude non-source files - accept all resource files, but only *.{java,scala} from source paths
  val onlySourceFiles: JarFileFilter = (p: Path, relPath: RelPath) => {
    p.last == "resources" || (relPath.ext == "scala" || relPath.ext == "java")
  }

  // Apply a sequence of file filters - only accept files which satisfy all filters.
  // We expect this to be curried, the resulting file filter passed to createJar()
  def forallFilters(fileFilters: Seq[JarFileFilter])(p: Path, relPath: RelPath): Boolean = {
    fileFilters.forall(f => f(p, relPath))
  }

  // We need to copy (and override) the `jar` and `docJar` targets so we can build
  //  single jars implementing our "all-in-one" policy.
  override def jar = T {
    createJar(
      allModuleClasspath().map(_.path).filter(exists),
      mainClass(),
      noDS_StoreFiles
    )
  }


  override def docJar = T {
    val outDir = T.ctx().dest

    val javadocDir = outDir / 'javadoc
    mkdir(javadocDir)

    val files = for{
      ref <- allModuleSources()
      if exists(ref.path)
      p <- (if (ref.path.isDir) ls.rec(ref.path) else Seq(ref.path))
      if (p.isFile && ((p.ext == "scala") || (p.ext == "java")))
    } yield p.toNIO.toString

    val pluginOptions = scalacPluginClasspath().map(pluginPathRef => s"-Xplugin:${pluginPathRef.path}")
    val options = Seq("-d", javadocDir.toNIO.toString, "-usejavacp") ++ pluginOptions ++ scalacOptions()

    if (files.nonEmpty) subprocess(
      "scala.tools.nsc.ScalaDoc",
      scalaCompilerClasspath().map(_.path) ++ compileClasspath().filter(_.path.ext != "pom").map(_.path),
      mainArgs = (files ++ options).toSeq
    )

    createJar(Agg(javadocDir), None, noDS_StoreFiles)(outDir)
  }

  def sourceJar = T {
    createJar((allModuleSources() ++ allModuleResources()).map(_.path).filter(exists), None, forallFilters(Seq(noDS_StoreFiles, onlySourceFiles)))
  }

  override def ivyDeps = Agg(
    ivy"com.github.scopt::scopt:3.6.0"
  ) ++ chiselDeps

  object test extends Tests {
    override def ivyDeps = Agg(
      ivy"org.scalatest::scalatest:3.0.1",
      ivy"org.scalacheck::scalacheck:1.13.4"
    )
    def testFrameworks = Seq("org.scalatest.tools.Framework")
  }

  // This shouldn't be necessary - mill is supposed to figure out the main class.
  def mainClass = Some("chisel3.Driver")

}
