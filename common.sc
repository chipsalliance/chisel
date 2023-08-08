import mill._
import mill.scalalib._

trait HasMacroAnnotations
  extends ScalaModule {

  override def scalacOptions = T {
    super.scalacOptions() ++ Agg("-Ymacro-annotations")
  }
}

trait MacrosModule
  extends ScalaModule
    with HasMacroAnnotations {
  def scalaReflectIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Some(scalaReflectIvy)
}

trait FirrtlModule
  extends ScalaModule
    with HasMacroAnnotations {
  def osLibModuleIvy: Dep

  def json4sIvy: Dep

  def dataclassIvy: Dep

  def commonTextIvy: Dep

  def scoptIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Agg(
    osLibModuleIvy,
    json4sIvy,
    dataclassIvy,
    commonTextIvy,
    scoptIvy
  )
}

trait SvsimModule
  extends ScalaModule {
}

trait SvsimUnitTestModule
  extends TestModule
    with ScalaModule
    with TestModule.ScalaTest {
  def svsimModule: SvsimModule

  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def moduleDeps = Seq(svsimModule)

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}

trait FirrtlUnitTestModule
  extends TestModule
    with ScalaModule
    with TestModule.ScalaTest {
  def firrtlModule: FirrtlModule

  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def moduleDeps = Seq(firrtlModule)

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}

trait CoreModule
  extends ScalaModule
    with HasMacroAnnotations {
  def firrtlModule: FirrtlModule

  def macrosModule: MacrosModule

  def osLibModuleIvy: Dep

  def upickleModuleIvy: Dep

  override def moduleDeps = super.moduleDeps ++ Seq(macrosModule, firrtlModule)

  override def ivyDeps = super.ivyDeps() ++ Agg(
    osLibModuleIvy,
    upickleModuleIvy
  )
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
    with HasChisel

trait ChiselModule
  extends ScalaModule
    with HasChiselPlugin
    with HasMacroAnnotations {
  def macrosModule: MacrosModule

  def svsimModule: SvsimModule

  def coreModule: CoreModule

  override def scalacPluginClasspath = T(super.scalacPluginClasspath() ++ Agg(pluginModule.jar()))

  override def moduleDeps = super.moduleDeps ++ Seq(macrosModule, coreModule, svsimModule)
}

trait HasChisel
  extends ScalaModule
    with HasChiselPlugin {
  def chiselModule: ChiselModule

  override def moduleDeps = super.moduleDeps ++ Some(chiselModule)
}

trait ChiselUnitTestModule
  extends TestModule
    with ScalaModule
    with HasChisel
    with HasMacroAnnotations
    with TestModule.ScalaTest {
  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}
