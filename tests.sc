import mill._
import mill.api.Result
import mill.scalalib._
import mill.scalalib.api.CompilationResult
import mill.util.Jvm

import java.util
import scala.jdk.StreamConverters.StreamHasToScala

trait SvsimUnitTestModule
  extends TestModule
    with ScalaModule
    with TestModule.ScalaTest {
  def svsimModule: common.SvsimModule

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
  def firrtlModule: common.FirrtlModule

  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def moduleDeps = Seq(firrtlModule)

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}

trait ChiselUnitTestModule
  extends TestModule
    with ScalaModule
    with common.HasChisel
    with common.HasMacroAnnotations
    with TestModule.ScalaTest {
  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}

trait CIRCTPanamaBinderModuleTestModule
  extends TestModule
    with ScalaModule
    with common.HasCIRCTPanamaBinderModule
    with common.HasMacroAnnotations
    with TestModule.ScalaTest {
  def scalatestIvy: Dep

  def scalacheckIvy: Dep

  override def defaultCommandName() = "test"

  override def ivyDeps = super.ivyDeps() ++ Agg(
    scalatestIvy,
    scalacheckIvy
  )
}

trait LitUtilityModule
  extends ScalaModule
    with common.HasCIRCTPanamaBinderModule
    with common.HasMacroAnnotations

trait LitModule
  extends Module {
  def crossScalaVersions: T[Seq[String]]
  def runClasspath: T[Seq[os.Path]]
  def pluginJars: T[Seq[os.Path]]
  def javaLibraryPath: T[Seq[os.Path]]
  def javaHome: T[os.Path]
  def chiselLitDir: T[os.Path]
  def litConfigIn: T[PathRef]
  def litConfig: T[PathRef] = T {
    os.write(
      T.dest / "lit.site.cfg.py",
      os.read(litConfigIn().path)
        .replaceAll("@CROSS_SCALA_VERSION@", crossScalaVersions().mkString(","))
        .replaceAll("@RUN_CLASSPATH@", runClasspath().mkString(","))
        .replaceAll("@OPT_PLUGIN_JARS@", pluginJars().mkString(","))
        .replaceAll("@JAVA_HOME@", javaHome().toString)
        .replaceAll("@JAVA_OPT_JAVA_LIBRARY_PATH@", javaLibraryPath().mkString(","))
        .replaceAll("@CHISEL_LIT_DIR@", chiselLitDir().toString)
      )
    PathRef(T.dest)
  }
  def run(args: String*) = T.command(os.proc("llvm-lit", litConfig().path).call(T.dest, stdout = os.ProcessOutput.Readlines(line => T.ctx().log.debug("[lit] " + line))))
 }
