// This file only records the recipe to

import mill._
import mill.scalalib._
import $ivy.`com.lihaoyi::mill-contrib-buildinfo:`
import mill.contrib.buildinfo.BuildInfo

private def majorScalaVersion(scalaVersion: String) = scalaVersion.split('.')(1).toInt

trait HasMacroAnnotations
  extends ScalaModule {
  def macroParadiseIvy: Option[Dep]

  def scalacPluginIvyDeps = super.scalacPluginIvyDeps() ++ macroParadiseIvy

  override def scalacOptions = T {
    if (scalaVersion() == 12) {
      require(macroParadiseIvy.isDefined, "macroParadiseIvy must be defined for Scala 2.12")
    }
    super.scalacOptions() ++
      (if (majorScalaVersion(scalaVersion()) == 13) Agg("-Ymacro-annotations") else Agg.empty[String])
  }
}

trait MacrosModule
  extends ScalaModule
    with HasMacroAnnotations {
  def scalaReflectIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Some(scalaReflectIvy)

  override def scalacPluginIvyDeps = super.scalacPluginIvyDeps() ++ macroParadiseIvy
}

trait CoreModule
  extends ScalaModule
    with HasMacroAnnotations
    with BuildInfo {
  def firrtlModule: Option[ScalaModule] = None

  def firrtlIvyDeps: Option[Dep] = None

  def macrosModule: MacrosModule

  def osLibModuleIvy: Dep

  def upickleModuleIvy: Dep

  override def moduleDeps = super.moduleDeps ++ Some(macrosModule) ++ firrtlModule

  override def ivyDeps = super.ivyDeps() ++ Agg(
    osLibModuleIvy,
    upickleModuleIvy
  ) ++ {
    require(
      firrtlModule.isDefined ^ firrtlIvyDeps.isDefined,
      "Either 'firrtlModule' or 'firrtlIvyDeps' should be defined in your build script."
    )
    firrtlIvyDeps
  }

  // BuildInfo
  override def buildInfoPackageName = Some("chisel3")

  def buildVersion = T("build-from-source")

  override def buildInfoMembers = T {
    Map(
      "buildInfoPackage" -> artifactName(),
      "version" -> buildVersion(),
      "scalaVersion" -> scalaVersion()
    )
  }

}

trait PluginModule
  extends ScalaModule {
  def scalaLibraryIvy: Dep

  def scalaReflectIvy: Dep

  def scalaCompilerIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Agg(scalaLibraryIvy, scalaReflectIvy, scalaCompilerIvy)
}

trait HasChiselPlugin
  extends ScalaModule {
  def pluginModule: PluginModule

  override def scalacOptions = T {
    super.scalacOptions() ++ Agg(s"-Xplugin:${pluginModule.jar().path}")
  }

  override def scalacPluginClasspath = T {
    super.scalacPluginClasspath() ++ Agg(
      pluginModule.jar()
    )
  }
}

trait StdLibModule
  extends ScalaModule
    with HasChiselPlugin {
  def chiselModule: ChiselModule

  override def moduleDeps = super.moduleDeps ++ Seq(chiselModule)
}

trait ChiselModule
  extends ScalaModule
    with HasChiselPlugin
    with HasMacroAnnotations {
  def macrosModule: MacrosModule

  def coreModule: CoreModule

  override def scalacPluginClasspath = T(super.scalacPluginClasspath() ++ Agg(pluginModule.jar()))

  override def moduleDeps = super.moduleDeps ++ Seq(macrosModule, coreModule)
}