import mill._
import mill.api.Result
import mill.scalalib._
import mill.scalalib.api.CompilationResult
import mill.util.Jvm

import java.util
import scala.jdk.StreamConverters.StreamHasToScala

trait LitUtilityModule extends ScalaModule with common.HasPanamaConverterModule with common.HasPanamaOMModule {
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
