import ammonite.ops._
import ammonite.ops.ImplicitWd._
import mill._
import mill.scalalib._
import mill.scalalib.publish._
import mill.eval.{Evaluator, Result}
import mill.define.Task
import mill.modules.Jvm._

import $file.CommonBuild

/** Utility types for changing a dependency between Ivy and Module (source) */
sealed trait IvyOrModuleDep
case class IvyDep(dep: Dep) extends IvyOrModuleDep
case class ModuleDep(dep: PublishModule) extends IvyOrModuleDep

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

val crossVersions = Seq("2.12.6", "2.11.12")

// Provide a managed dependency on X if -DXVersion="" is supplied on the command line.
val defaultVersions = Map("firrtl" -> "1.2-SNAPSHOT")

def getVersion(dep: String, org: String = "edu.berkeley.cs") = {
  val version = sys.env.getOrElse(dep + "Version", defaultVersions(dep))
  ivy"$org::$dep:$version"
}

// Define the common chisel module.
trait CommonChiselModule extends SbtModule {
  // Normally defined in CrossSbtModule, our submodules don't have it by default
  def crossScalaVersion: String
  // This build uses an ivy dependency but allows overriding with a module (source) dependency
  def firrtlDep: IvyOrModuleDep
  override def scalacOptions = chiselCompileOptions.scalacOptions ++ CommonBuild.scalacOptionsVersion(crossScalaVersion)
  override def javacOptions = CommonBuild.javacOptionsVersion(crossScalaVersion)
  val macroPlugins = Agg(ivy"org.scalamacros:::paradise:2.1.0")
  def scalacPluginIvyDeps = macroPlugins
  def compileIvyDeps = macroPlugins
  def chiselDeps = firrtlDep match {
    case IvyDep(dep) => Agg(dep)
    case ModuleDep(_) => Agg()
  }
  override def ivyDeps = T { chiselDeps }
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

// Make this available to external tools.
object chisel3 extends Cross[ChiselTopModule](crossVersions: _*) {
  def defaultVersion(ev: Evaluator) = T.command{
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

class ChiselTopModule(val crossScalaVersion: String) extends AbstractChiselModule {
  // This build uses an ivy dependency but allows overriding with a module (source) dependency
  def firrtlDep: IvyOrModuleDep = IvyDep(getVersion("firrtl"))
}

trait AbstractChiselModule extends PublishChiselModule with CommonBuild.BuildInfo with CrossSbtModule { top =>

  // If would be nice if we didn't need to do this, but PublishModule may only be dependent on
  //  other PublishModules.
  trait UnpublishedChiselModule extends PublishChiselModule


  object coreMacros extends UnpublishedChiselModule {
    def crossScalaVersion = top.crossScalaVersion
    def scalaVersion = crossScalaVersion
    def firrtlDep: IvyOrModuleDep = top.firrtlDep
  }

  object chiselFrontend extends UnpublishedChiselModule {
    def crossScalaVersion = top.crossScalaVersion
    def scalaVersion = crossScalaVersion
    def moduleDeps = Seq(coreMacros) ++ (firrtlDep match {
      case ModuleDep(dep) => Seq(dep)
      case IvyDep(_) => Seq()
    })
    def firrtlDep: IvyOrModuleDep = top.firrtlDep
  }

  override def moduleDeps = Seq(coreMacros, chiselFrontend)

  // This submodule is unrooted - its source directory is in the top level directory.
  override def millSourcePath = super.millSourcePath / ammonite.ops.up

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

  // Override publishXmlDeps so we don't include dependencies on our sub-modules.
  override def publishXmlDeps = T.task {
    val ivyPomDeps = ivyDeps().map(resolvePublishDependency().apply(_))
    ivyPomDeps
  }

  // We need to copy (and override) the `jar` and `docJar` targets so we can build
  //  single jars implementing our "all-in-one" policy.
  override def jar = T {
    createJar(
      allModuleClasspath().map(_.path).filter(exists),
      mainClass()
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

    if (files.nonEmpty) runSubprocess(
      "scala.tools.nsc.ScalaDoc",
      scalaCompilerClasspath().map(_.path) ++ compileClasspath().filter(_.path.ext != "pom").map(_.path),
      mainArgs = (files ++ options).toSeq
    )

    createJar(Agg(javadocDir), None)(outDir)
  }

  def sourceJar = T {
    createJar((allModuleSources() ++ allModuleResources()).map(_.path).filter(exists), None)
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

  // This is required for building a library, but not for a `run` target.
  // In the latter case, mill will determine this on its own.
  def mainClass = Some("chisel3.Driver")

  override def buildInfoMembers = T {
    Map[String, String](
      "buildInfoPackage" -> artifactName(),
      "version" -> publishVersion(),
      "scalaVersion" -> scalaVersion()
    )
  }
}
