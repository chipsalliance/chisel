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

trait CIRCTModule
  extends Module {
  def circtSourcePath: T[PathRef]
  def llvmSourcePath: T[PathRef]
  def installDirectory: T[PathRef] = T(PathRef(T.dest))

  def parallelCounts: T[Int] = T(java.lang.Runtime.getRuntime.availableProcessors())

  def install = T {
    os.proc("ninja", s"-j${parallelCounts()}", "install").call(cmake())
  }

  def cmake = T.persistent {
    os.proc(
      "cmake",
      "-S", llvmSourcePath().path / "llvm",
      "-B", T.dest,
      "-G", "Ninja",
      s"-DCMAKE_INSTALL_PREFIX=${installDirectory().path}",
      "-DCMAKE_BUILD_TYPE=Release",
      "-DLLVM_ENABLE_PROJECTS=mlir",
      "-DLLVM_TARGETS_TO_BUILD=X86",
      "-DLLVM_ENABLE_ASSERTIONS=OFF",
      "-DLLVM_BUILD_EXAMPLES=OFF",
      "-DLLVM_INCLUDE_EXAMPLES=OFF",
      "-DLLVM_INCLUDE_TESTS=OFF",
      "-DLLVM_INSTALL_UTILS=OFF",
      "-DLLVM_ENABLE_OCAMLDOC=OFF",
      "-DLLVM_ENABLE_BINDINGS=OFF",
      "-DLLVM_CCACHE_BUILD=OFF",
      "-DLLVM_BUILD_TOOLS=OFF",
      "-DLLVM_OPTIMIZED_TABLEGEN=ON",
      "-DLLVM_USE_SPLIT_DWARF=ON",
      "-DLLVM_BUILD_LLVM_DYLIB=OFF",
      "-DLLVM_LINK_LLVM_DYLIB=OFF",
      "-DLLVM_EXTERNAL_PROJECTS=circt",
      "-DBUILD_SHARED_LIBS=ON",
      s"-DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=${circtSourcePath().path}"
    ).call(T.dest)
    T.dest
  }
}

/** Module to extract header from CIRCT. */
trait CIRCTPanamaModule
  extends JavaModule {
  def circtModule: CIRCTModule
  def circtInstallDirectory: T[PathRef] = T {
    circtModule.install()
    circtModule.installDirectory()
  }
  def javacVersion = T.input {
    val version = os.proc("javac", "-version").call().out.text.split(' ').last.split('.').head.toInt
    require(version >= 19, "Java 19 or higher is required")
    version
  }

  override def javacOptions: T[Seq[String]] = {
    Seq("--enable-preview", "--source", javacVersion().toString)
  }

  def jextractTarGz = T.persistent {
    val f = T.dest / "jextract.tar.gz"
    if (!os.exists(f))
      modules.Util.download(s"https://download.java.net/java/early_access/jextract/2/openjdk-19-jextract+2-3_linux-x64_bin.tar.gz", os.rel / "jextract.tar.gz")
    PathRef(f)
  }

  def jextract = T.persistent {
    os.proc("tar", "xvf", jextractTarGz().path).call(T.dest)
    PathRef(T.dest / "jextract-19" / "bin" / "jextract")
  }

  // Task to generate all possible bindings
  def dumpAllIncludes = T {
    val f = os.temp()
    os.proc(
      jextract().path,
      circtInstallDirectory().path / "include" / "circt-c" / "Dialect" / "FIRRTL.h",
      "-I", circtInstallDirectory().path / "include",
      "--dump-includes", f
    ).call()
    os.read.lines(f).filter(s => s.nonEmpty && !s.startsWith("#"))
  }

  def includeFunctions = T {
    Seq(
      "firrtlCreateContext",
      "firrtlDestroyContext",
      "firrtlSetErrorHandler",
      "firrtlVisitCircuit",
      "firrtlVisitModule",
      "firrtlVisitExtModule",
      "firrtlVisitParameter",
      "firrtlVisitPort",
      "firrtlVisitStatement",
      "firrtlExportFirrtl",
      "firrtlDestroyString",
    )
  }

  def includeMacros = T {
    Seq.empty[String]
  }

  def includeStructs = T {
    Seq(
      "MlirStringRef",
      "FirrtlContext",
      "FirrtlParameterInt",
      "FirrtlParameterDouble",
      "FirrtlParameterString",
      "FirrtlParameterRaw",
      "FirrtlParameter",
      "FirrtlTypeUInt",
      "FirrtlTypeSInt",
      "FirrtlTypeClock",
      "FirrtlTypeReset",
      "FirrtlTypeAsyncReset",
      "FirrtlTypeAnalog",
      "FirrtlTypeVector",
      "FirrtlTypeBundleField",
      "FirrtlTypeBundle",
      "FirrtlType",
      "FirrtlStatementAttachOperand",
      "FirrtlStatementAttach",
      "FirrtlStatement",
    )
  }

  def includeTypedefs = T {
    Seq(
      "FirrtlStringRef",
      "FirrtlErrorHandler",
      // enums (FIXME)
      "FirrtlPortDirection",
      "FirrtlParameterKind",
      "FirrtlTypeKind",
      "FirrtlStatementKind",
    )
  }

  def includeUnions = T {
    Seq(
      "FirrtlParameterUnion",
      "FirrtlTypeUnion",
      "FirrtlStatementUnion",
    )
  }

  def includeVars = T {
    Seq.empty[String]
  }

  override def generatedSources: T[Seq[PathRef]] = T {
    os.proc(
      Seq(
        jextract().path.toString,
        (circtInstallDirectory().path / "include" / "circt-c" / "Dialect" / "FIRRTL.h").toString,
        "-I", (circtInstallDirectory().path / "include").toString,
        "-t", "org.llvm.circt.firrtl",
        "-l", "CIRCTCAPIFIRRTL",
        "--header-class-name", "CIRCTCAPIFIRRTL",
        "--source",
        "--output", T.dest.toString
      ) ++ includeFunctions().flatMap(f => Seq("--include-function", f)) ++
        includeMacros().flatMap(f => Seq("--include-macro", f)) ++
        includeStructs().flatMap(f => Seq("--include-struct", f)) ++
        includeTypedefs().flatMap(f => Seq("--include-typedef", f)) ++
        includeUnions().flatMap(f => Seq("--include-union", f)) ++
        includeVars().flatMap(f => Seq("--include-var", f))
    ).call()
    Lib.findSourceFiles(os.walk(T.dest).map(PathRef(_)), Seq("java")).distinct.map(PathRef(_))
  }

  // Zinc doesn't happy with the --enable-preview flag, so we work around it
  // since there is no incremental compilation needs
  final override def compile: T[mill.scalalib.api.CompilationResult] = T {
    os.proc(Seq("javac", "-d", T.dest.toString) ++ javacOptions() ++ allSourceFiles().map(_.path.toString)).call(T.dest)
    mill.scalalib.api.CompilationResult(os.root, PathRef(T.dest))
  }
}

/** Scala Module to implement chisel3.compiler.CompilerApi */
trait ChiselCIRCTPanamaModule
  extends ScalaModule {
  def coreModule: CoreModule
  def circtPanamaModule: CIRCTPanamaModule
  override def moduleDeps = super.moduleDeps ++ Seq(coreModule, circtPanamaModule)
  def circtInstallDirectory: T[PathRef] = T(circtPanamaModule.circtInstallDirectory())
  override def forkArgs: T[Seq[String]] = {
    super.forkArgs() ++ Seq(
      "--enable-native-access=ALL-UNNAMED",
      "--enable-preview",
      s"-Djava.library.path=${circtInstallDirectory().path / "lib"}"
    )
  }
}