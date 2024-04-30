import mill._
import mill.scalalib._

trait HasMacroAnnotations extends ScalaModule {

  override def scalacOptions = T {
    super.scalacOptions() ++ Agg("-Ymacro-annotations")
  }
}

trait MacrosModule extends ScalaModule with HasMacroAnnotations {
  def scalaReflectIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Some(scalaReflectIvy)
}

trait FirrtlModule extends ScalaModule with HasMacroAnnotations {
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

trait SvsimModule extends ScalaModule {}

trait CoreModule extends ScalaModule with HasMacroAnnotations {
  def firrtlModule: FirrtlModule

  def macrosModule: MacrosModule

  def osLibModuleIvy: Dep

  def upickleModuleIvy: Dep

  def mainargsModuleIvy: Dep

  def firtoolResolverModuleIvy: Dep

  override def moduleDeps = super.moduleDeps ++ Seq(macrosModule, firrtlModule)

  override def ivyDeps = super.ivyDeps() ++ Agg(
    osLibModuleIvy,
    upickleModuleIvy,
    mainargsModuleIvy,
    firtoolResolverModuleIvy
  )
}

trait PluginModule extends ScalaModule {
  def scalaLibraryIvy: Dep

  def scalaReflectIvy: Dep

  def scalaCompilerIvy: Dep

  override def ivyDeps = super.ivyDeps() ++ Agg(scalaLibraryIvy, scalaReflectIvy, scalaCompilerIvy)
}

trait HasChiselPlugin extends ScalaModule {
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

trait StdLibModule extends ScalaModule with HasChisel

trait ChiselModule extends ScalaModule with HasChiselPlugin with HasMacroAnnotations {
  def macrosModule: MacrosModule

  def svsimModule: SvsimModule

  def coreModule: CoreModule

  override def scalacPluginClasspath = T(super.scalacPluginClasspath() ++ Agg(pluginModule.jar()))

  override def moduleDeps = super.moduleDeps ++ Seq(macrosModule, coreModule, svsimModule)
}

trait HasChisel extends ScalaModule with HasChiselPlugin {
  def chiselModule: ChiselModule

  override def moduleDeps = super.moduleDeps ++ Some(chiselModule)
}

trait HasJextractGeneratedSources extends JavaModule {
  def includePaths: T[Seq[PathRef]]

  def libraryPaths: T[Seq[PathRef]]

  def header: T[PathRef]

  def includeFunctions: T[Seq[String]]

  def includeConstants: T[Seq[String]]

  def includeStructs: T[Seq[String]]

  def includeTypedefs: T[Seq[String]]

  def includeUnions: T[Seq[String]]

  def includeVars: T[Seq[String]]

  def linkLibraries: T[Seq[String]]

  def target: T[String]

  def headerClassName: T[String]

  def dumpAllIncludes = T {
    val f = os.temp()
    os.proc(
      Seq("jextract", header().path.toString)
        ++ includePaths().flatMap(p => Seq("-I", p.path.toString))
        ++ Seq("--dump-includes", f.toString)
    ).call()
    os.read.lines(f).filter(s => s.nonEmpty && !s.startsWith("#"))
  }

  override def generatedSources: T[Seq[PathRef]] = T {
    super.generatedSources() ++ {
      // @formatter:off
      os.proc(
        Seq("jextract", header().path.toString)
          ++ includePaths().flatMap(p => Seq("-I", p.path.toString))
          ++ Seq(
          "-t", target(),
          "--header-class-name", headerClassName(),
          "--source",
          "--output", T.dest.toString
        ) ++ includeFunctions().flatMap(f => Seq("--include-function", f)) ++
          includeConstants().flatMap(f => Seq("--include-constant", f)) ++
          includeStructs().flatMap(f => Seq("--include-struct", f)) ++
          includeTypedefs().flatMap(f => Seq("--include-typedef", f)) ++
          includeUnions().flatMap(f => Seq("--include-union", f)) ++
          includeVars().flatMap(f => Seq("--include-var", f)) ++
          linkLibraries().flatMap(l => Seq("-l", l))
      ).call(T.dest)
      // @formatter:on
      Lib
        .findSourceFiles(os.walk(T.dest).map(PathRef(_)), Seq("java"))
        .distinct
        .map(PathRef(_))
    }
  }

  override def javacOptions = T(super.javacOptions() ++ Seq("--enable-preview", "--release", "21"))
}

// Java Codegen for all declared functions.
// All of these functions are not private API which is subject to change.
trait CIRCTPanamaBindingModule extends HasJextractGeneratedSources {

  def includeConstants =
    T.input(os.read.lines(millSourcePath / "includeConstants.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def includeFunctions =
    T.input(os.read.lines(millSourcePath / "includeFunctions.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def includeStructs =
    T.input(os.read.lines(millSourcePath / "includeStructs.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def includeTypedefs =
    T.input(os.read.lines(millSourcePath / "includeTypedefs.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def includeUnions =
    T.input(os.read.lines(millSourcePath / "includeUnions.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def includeVars =
    T.input(os.read.lines(millSourcePath / "includeVars.txt").filter(s => s.nonEmpty && !s.startsWith("#")))
  def linkLibraries =
    T.input(os.read.lines(millSourcePath / "linkLibraries.txt").filter(s => s.nonEmpty && !s.startsWith("#")))

  def target = T("org.llvm.circt")
  def headerClassName = T("CAPI")
}

trait HasCIRCTPanamaBindingModule extends JavaModule {
  def circtPanamaBindingModule: CIRCTPanamaBindingModule

  override def moduleDeps = super.moduleDeps ++ Some(circtPanamaBindingModule)

  override def javacOptions = T(super.javacOptions() ++ Seq("--enable-preview", "--release", "21"))

  override def forkArgs: T[Seq[String]] = T(
    super.forkArgs() ++ Seq("--enable-native-access=ALL-UNNAMED", "--enable-preview")
      ++ circtPanamaBindingModule
        .libraryPaths()
        .map(p => s"-Djava.library.path=${p.path}")
  )
}

// The Scala API for PanamaBinding, API here is experimentally public to all developers
trait PanamaLibModule extends ScalaModule with HasCIRCTPanamaBindingModule

trait HasPanamaLibModule extends ScalaModule with HasCIRCTPanamaBindingModule {
  def panamaLibModule: PanamaLibModule

  def circtPanamaBindingModule = panamaLibModule.circtPanamaBindingModule

  override def moduleDeps = super.moduleDeps ++ Some(panamaLibModule)
}

trait PanamaOMModule extends ScalaModule with HasPanamaLibModule

trait HasPanamaOMModule extends ScalaModule with HasCIRCTPanamaBindingModule {
  def panamaOMModule: PanamaOMModule

  def circtPanamaBindingModule = panamaOMModule.circtPanamaBindingModule

  override def moduleDeps = super.moduleDeps ++ Some(panamaOMModule)
}

trait PanamaConverterModule extends ScalaModule with HasPanamaOMModule with HasChisel

trait HasPanamaConverterModule extends ScalaModule with HasCIRCTPanamaBindingModule with HasChisel {
  def panamaConverterModule: PanamaConverterModule

  def circtPanamaBindingModule = panamaConverterModule.circtPanamaBindingModule

  override def chiselModule = panamaConverterModule.chiselModule

  override def pluginModule = panamaConverterModule.pluginModule

  override def moduleDeps = super.moduleDeps ++ Some(panamaConverterModule)
}
