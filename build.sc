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
  val osLib = ivy"com.lihaoyi::os-lib:0.8.1"
  val upickle = ivy"com.lihaoyi::upickle:2.0.0"
  val macroParadise = ivy"org.scalamacros:::paradise:2.1.1"
  val scalatest = ivy"org.scalatest::scalatest:3.2.14"
  val scalacheck = ivy"org.scalatestplus::scalacheck-1-14:3.2.2.0"
  val json4s = ivy"org.json4s::json4s-native:4.0.6"
  val dataclass = ivy"io.github.alexarchambault::data-class:0.2.5"
  val commonText = ivy"org.apache.commons:commons-text:1.10.0"
  val scopt = ivy"com.github.scopt::scopt:3.7.1"

  def scalaReflect(scalaVersion: String) = ivy"org.scala-lang:scala-reflect:$scalaVersion"

  def scalaCompiler(scalaVersion: String) = ivy"org.scala-lang:scala-compiler:$scalaVersion"

  def scalaLibrary(scalaVersion: String) = ivy"org.scala-lang:scala-library:$scalaVersion"
}
private def majorScalaVersion(scalaVersion: String) = scalaVersion.split('.')(1).toInt

object firrtl extends mill.Cross[firrtl](v.scalaCrossVersions: _*)

class firrtl(val crossScalaVersion: String)
  extends common.FirrtlModule
    with CrossSbtModule
    with ScalafmtModule {
  def macroParadiseIvy: Option[Dep] = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None

  def osLibModuleIvy = v.osLib

  def json4sIvy = v.json4s

  def dataclassIvy = v.dataclass

  def commonTextIvy = v.commonText

  def scoptIvy = v.scopt
}

object firrtlut extends mill.Cross[firrtlUnitTest](v.scalaCrossVersions: _*)

class firrtlUnitTest(val crossScalaVersion: String)
  extends common.FirrtlUnitTestModule
    with CrossModuleBase
    with ScalafmtModule {
  override def millSourcePath = firrtl(crossScalaVersion).millSourcePath

  def firrtlModule = firrtl(crossScalaVersion)

  def scalatestIvy = v.scalatest

  def scalacheckIvy = v.scalacheck

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "src" / "test")) ++
      matchingVersions(crossScalaVersion).map(s =>
        PathRef(millSourcePath / "src" / "test" / s"scala-$s")
      )
  }
}

object macros extends mill.Cross[macros](v.scalaCrossVersions: _*)

class macros(val crossScalaVersion: String)
  extends common.MacrosModule
    with CrossSbtModule
    with ScalafmtModule {
  def scalaReflectIvy = v.scalaReflect(crossScalaVersion)

  def macroParadiseIvy: Option[Dep] = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None
}

object core extends mill.Cross[core](v.scalaCrossVersions: _*)

class core(val crossScalaVersion: String)
  extends common.CoreModule
    with CrossSbtModule
    with ScalafmtModule {
  def firrtlModule = firrtl(crossScalaVersion)

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

object chisel extends mill.Cross[chisel](v.scalaCrossVersions: _*)

class chisel(val crossScalaVersion: String)
  extends common.ChiselModule
    with CrossSbtModule
    with ScalafmtModule {
  override def millSourcePath = super.millSourcePath / os.up

  def macrosModule = macros(crossScalaVersion)

  def coreModule = core(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)

  def macroParadiseIvy = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None
}

object chiselut extends mill.Cross[chiselUnitTest](v.scalaCrossVersions: _*)

class chiselUnitTest(val crossScalaVersion: String)
  extends common.ChiselUnitTestModule
    with CrossModuleBase
    with ScalafmtModule {
  override def millSourcePath = chisel(crossScalaVersion).millSourcePath

  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)

  def scalatestIvy = v.scalatest

  def scalacheckIvy = v.scalacheck

  def macroParadiseIvy: Option[Dep] = if (majorScalaVersion(crossScalaVersion) < 13) Some(v.macroParadise) else None

  override def sources = T.sources {
    Seq(PathRef(millSourcePath / "src" / "test")) ++
      matchingVersions(crossScalaVersion).map(s =>
        PathRef(millSourcePath / "src" / "test" / s"scala-$s")
      )
  }
}


object stdlib extends mill.Cross[stdlib](v.scalaCrossVersions: _*)

class stdlib(val crossScalaVersion: String)
  extends common.StdLibModule
    with CrossSbtModule
    with ScalafmtModule {
  def chiselModule = chisel(crossScalaVersion)

  def pluginModule = plugin(crossScalaVersion)
}
