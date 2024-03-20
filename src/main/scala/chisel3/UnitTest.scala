package chisel3.test

import chisel3._
import chisel3.stage.ChiselGeneratorAnnotation
import circt.stage.{ChiselStage, CIRCTTarget, CIRCTTargetAnnotation}
import java.io.File
import java.util.jar.JarFile
import scala.collection.JavaConverters._

trait RootTest

/** Helper to discover all subtypes of `RootTest` in the class path, and call
 *  their constructors (if they are a class) or ensure that the singleton is
 *  constructed (if they are an object). */
object DiscoverTests {

  /** Discover and construct all tests in the classpath. */
  def apply(): Unit = classpath().foreach(discoverFile(_))

  /** Return the a sequence of files or directories on the classpath. */
  private def classpath(): Iterable[File] = System
      .getProperty("java.class.path")
      .split(File.pathSeparator)
      .map(s => if (s.trim.length == 0) "." else s)
      .map(new File(_))

  /** Discover all tests in a given file. If this is a JAR file, looks through
   *  its contents and tries to find its classes. */
  private def discoverFile(file: File): Unit = {
    if (file.getPath.toLowerCase.endsWith(".jar")) {
      val jarFile = new java.util.jar.JarFile(file)
      jarFile.entries.asScala.foreach { jarEntry =>
        val name = jarEntry.getName
        if (!jarEntry.isDirectory && name.endsWith(".class"))
          discoverClass(name.stripPrefix("/").stripSuffix(".class").replace('/', '.'))
      }
    }
  }

  /** Load the given class and check whether it is a subtype of `RootTest`. If
   *  it is, call its constructor if it is a class, or ensure it is constructed
   *  if it is an object. */
  private def discoverClass(className: String): Unit = {
    val clazz = try {
      classOf[RootTest].getClassLoader.loadClass(className)
    } catch {
      case e: ClassNotFoundException => return
      case e: NoClassDefFoundError => return
      case e: ClassCastException => return
      case e: UnsupportedClassVersionError => return
    }

    // Check if it is a subtype of `RootTest` (and also not the definition of
    // `RootTest` itself).
    if (clazz == classOf[RootTest] || !classOf[RootTest].isAssignableFrom(clazz))
      return
    println(f"Building ${clazz}")

    // Ensure singleton objects are constructed.
    try {
      clazz.getField("MODULE$").get(null)
    } catch {
      case e: NoSuchFieldException => ()
    }

    // Call the constructor for test classes.
    try {
      clazz.getConstructor().newInstance()
    } catch {
      case e: NoSuchMethodException => ()
      case e: IllegalAccessException => ()
    }
  }
}

/** A dummy empty Chisel module that discovers and builds all `RootTest`
 *  subtypes on the classpath. This can be used in conjunction with
 *  `ChiselStage` to build all unit tests. */
class UnitTests extends RawModule {
  DiscoverTests()
}

/** Entry point to collect all `RootTest` subtypes in the classpath, run them,
 *  and collect the resulting hardware. */
object build {
  def main(args: Array[String]): Unit = {
    val stage = new circt.stage.ChiselStage
    stage.execute(args, Seq(
      ChiselGeneratorAnnotation(() => new UnitTests),
      CIRCTTargetAnnotation(CIRCTTarget.CHIRRTL),
    ))
  }
}
