import mill._
import mill.scalalib._
import mill.api.Result
import mill.scalalib._
import mill.scalalib.api.CompilationResult
import mill.util.Jvm
import mill.scalalib.scalafmt._

import java.util
import scala.jdk.StreamConverters.StreamHasToScala

object utils extends Module {
  val architecture = System.getProperty("os.arch")
  val operationSystem = System.getProperty("os.name")

  val mac = operationSystem.toLowerCase.startsWith("mac")
  val linux = operationSystem.toLowerCase.startsWith("linux")
  val windows = operationSystem.toLowerCase.startsWith("win")
  val amd64 = architecture.matches("^(x8664|amd64|ia32e|em64t|x64|x86_64)$")
  val aarch64 = architecture.equals("aarch64") | architecture.startsWith("armv8")

  val firtoolVersion = {
    val j = _root_.upickle.default.read[Map[String, String]](os.read(millSourcePath / os.up / "etc" / "circt.json"))
    j("version").stripPrefix("firtool-")
  }

  def circt(version: String, os: String, platform: String) =
    s"https://github.com/llvm/circt/releases/download/firtool-${version}/circt-full-shared-${os}-${platform}.tar.gz"

  // 21, 1-2, {linux-x64, macos-x64, windows-x64}
  // 22, 1-2, {linux-x64, macos-aarch64, macos-x64, windows-x64}
  def jextract(jdkVersion: Int, jextractVersion: String, os: String, platform: String) =
    s"https://download.java.net/java/early_access/jextract/22/6/openjdk-${jdkVersion}-jextract+${jextractVersion}_${os}-${platform}_bin.tar.gz"

  // use T.persistent to avoid download repeatedly
  def circtInstallDir: T[os.Path] = T.persistent {
    T.ctx().env.get("CIRCT_INSTALL_PATH") match {
      case Some(dir) => os.Path(dir)
      case None =>
        T.ctx().log.info("Use CIRCT_INSTALL_PATH to vendor circt")
        val tarPath = T.dest / "circt.tar.gz"
        if (!os.exists(tarPath)) {
          val url = circt(
            firtoolVersion,
            if (linux) "linux" else if (mac) "macos" else throw new Exception("unsupported os"),
            // circt does not yet publish for macos-aarch64, use x64 for now
            if (amd64 || mac) "x64" else throw new Exception("unsupported arch")
          )
          T.ctx().log.info(s"Downloading circt from ${url}")
          mill.util.Util.download(url, os.rel / "circt.tar.gz")
          T.ctx().log.info(s"Download Successfully")
        }
        os.proc("tar", "xvf", tarPath, "--strip-components=1").call(T.dest)
        T.dest
    }
  }

  // use T.persistent to avoid download repeatedly
  def jextractInstallDir: T[os.Path] = T.persistent {
    T.ctx().env.get("JEXTRACT_INSTALL_PATH") match {
      case Some(dir) => os.Path(dir)
      case None =>
        T.ctx().log.info("Use JEXTRACT_INSTALL_PATH to vendor jextract")
        val tarPath = T.dest / "jextract.tar.gz"
        if (!os.exists(tarPath)) {
          val url = jextract(
            22,
            "6-47",
            if (linux) "linux" else if (mac) "macos" else throw new Exception("unsupported os"),
            // There is no macos-aarch64 for jextract 21, use x64 for now
            if (amd64) "x64" else if (aarch64) "aarch64" else throw new Exception("unsupported arch")
          )
          T.ctx().log.info(s"Downloading jextract from ${url}")
          mill.util.Util.download(url, os.rel / "jextract.tar.gz")
          T.ctx().log.info(s"Download Successfully")
        }
        os.proc("tar", "xvf", tarPath, "--strip-components=1").call(T.dest)
        T.dest
    }
  }
}

trait HasJextractGeneratedSources extends JavaModule {

  def jextractBinary: T[os.Path]

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
      Seq(jextractBinary().toString, header().path.toString)
        ++ includePaths().flatMap(p => Seq("-I", p.path.toString))
        ++ Seq("--dump-includes", f.toString)
    ).call()
    os.read.lines(f).filter(s => s.nonEmpty && !s.startsWith("#"))
  }

  override def generatedSources: T[Seq[PathRef]] = T {
    super.generatedSources() ++ {
      // @formatter:off
      os.proc(
        Seq(jextractBinary().toString, header().path.toString)
          ++ includePaths().flatMap(p => Seq("-I", p.path.toString))
          ++ Seq(
            "-t", target(),
            "--header-class-name", headerClassName(),
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
      Seq(PathRef(T.dest))
    }
  }

  override def javacOptions = T(super.javacOptions() ++ Seq("--release", "22"))
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
  //
  override def javacOptions = T(super.javacOptions() ++ Seq("--release", "22"))

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

trait PanamaConverterModule extends ScalaModule with HasPanamaOMModule

trait HasPanamaConverterModule extends ScalaModule with HasCIRCTPanamaBindingModule {
  def panamaConverterModule: PanamaConverterModule

  def circtPanamaBindingModule = panamaConverterModule.circtPanamaBindingModule

  override def moduleDeps = super.moduleDeps ++ Some(panamaConverterModule)
}

trait PanamaOM extends PanamaOMModule with CrossModuleBase with ScalafmtModule

trait LitUtilityModule extends ScalaModule with HasPanamaConverterModule with HasPanamaOMModule {
  override def scalacOptions = T { Seq("-Ymacro-annotations") }
  override def circtPanamaBindingModule = panamaConverterModule.circtPanamaBindingModule
}

trait LitModule extends Module {
  def scalaVersion:    T[String]
  def runClasspath:    T[Seq[os.Path]]
  def pluginJars:      T[Seq[os.Path]]
  def javaLibraryPath: T[Seq[os.Path]]
  def javaHome:        T[os.Path]
  def chiselLitDir:    T[os.Path]
  def litConfigIn:     T[PathRef]
  def litConfig: T[PathRef] = T {
    os.write(
      T.dest / "lit.site.cfg.py",
      os.read(litConfigIn().path)
        .replaceAll("@SCALA_VERSION@", scalaVersion())
        .replaceAll("@RUN_CLASSPATH@", runClasspath().mkString(","))
        .replaceAll("@SCALA_PLUGIN_JARS@", pluginJars().mkString(","))
        .replaceAll("@JAVA_HOME@", javaHome().toString)
        .replaceAll("@JAVA_LIBRARY_PATH@", javaLibraryPath().mkString(","))
        .replaceAll("@CHISEL_LIT_DIR@", chiselLitDir().toString)
    )
    PathRef(T.dest)
  }
  def run(args: String*) = T.command(
    os.proc("lit", litConfig().path)
      .call(T.dest, stdout = os.ProcessOutput.Readlines(line => T.ctx().log.info("[lit] " + line)))
  )
}
