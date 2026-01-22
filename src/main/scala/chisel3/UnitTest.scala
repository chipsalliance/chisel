// SPDX-License-Identifier: Apache-2.0

package chisel3.test

import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.Definition
import chisel3.RawModule
import java.io.{File, IOException}
import java.net.URLClassLoader
import java.util.jar.JarFile
import scala.collection.JavaConverters._

/** All classes and objects marked as [[UnitTest]] are automatically
  * discoverable by the `DiscoverUnitTests` helper.
  */
trait UnitTest

/** Helper to discover all subtypes of [[UnitTest]] in the class path, and call
  * their constructors (if they are a class) or ensure that the singleton is
  * constructed (if they are an object).
  *
  * This code is loosely based on the test suite discovery in scalatest, which
  * performs the same scan over the classpath JAR files and directories, and
  * guesses class names based on the encountered directory structure.
  */
private[chisel3] object DiscoverUnitTests {

  /** The callback invoked for each unit test class name and unit test
    * constructor.
    */
  type Callback = (String, () => Unit) => Unit

  /** Discover all tests in the classpath or runpath and call `cb` for each. */
  def apply(cb: Callback, runpath: Seq[String]): Unit = {
    // Determine the list of files to scan for tests. This is either the
    // user-provided runpath if non-empty, or the classpath otherwise.
    val files =
      if (runpath.nonEmpty)
        runpath.map(new File(_))
      else
        classpath()

    // Construct a class loader that we use during test discovery.
    val loader = {
      // Map the runpath entries to URLs, and disable caching such that reruns
      // will reload classes.
      val urls = runpath.map(new File(_).toURI.toURL).toArray
      urls.foreach { url =>
        try {
          url.openConnection.setDefaultUseCaches(false)
        } catch {
          case e: IOException => // just ignore these
        }
      }

      // Create a class loader that first visits the provided runpath URLs, and
      // then falls back to the default class loader.
      new URLClassLoader(urls, classOf[UnitTest].getClassLoader)
    }

    // Discover tests.
    files.foreach(discoverFile(_, cb, loader))
  }

  /** Return the a sequence of files or directories on the classpath. */
  private def classpath(): Iterable[File] = System
    .getProperty("java.class.path")
    .split(File.pathSeparator)
    .map(s => if (s.trim.length == 0) "." else s)
    .map(new File(_))

  /** Discover all tests in a given file. If this is a JAR file, looks through
    * its contents and tries to find its classes.
    */
  private def discoverFile(file: File, cb: Callback, loader: ClassLoader): Unit = file match {
    // Unzip JAR files and process the class files they contain.
    case _ if file.getPath.toLowerCase.endsWith(".jar") =>
      val jarFile = new java.util.jar.JarFile(file)
      jarFile.entries.asScala.foreach { jarEntry =>
        val name = jarEntry.getName
        if (!jarEntry.isDirectory && name.endsWith(".class"))
          discoverClass(pathToClassName(name), cb, loader)
      }

    // Recursively collect any class files contained in directories.
    case _ if file.isDirectory =>
      def visit(prefix: String, file: File): Unit = {
        val name = prefix + "/" + file.getName
        if (file.isDirectory) {
          for (entry <- file.listFiles)
            visit(name, entry)
        } else if (name.endsWith(".class")) {
          discoverClass(pathToClassName(name), cb, loader)
        }
      }
      for (entry <- file.listFiles)
        visit("", entry)

    // Ignore any other files that aren't directories.
    case _ => ()
  }

  /** Convert a file path to a class */
  private def pathToClassName(path: String): String =
    path.replace('/', '.').replace('\\', '.').stripPrefix(".").stripSuffix(".class")

  /** Load the given class and check whether it is a subtype of [[UnitTest]]. If
    * it is, call the user-provided callback with a function that either calls
    * the loaded class' constructor or ensures the loaded object is constructed.
    */
  private def discoverClass(className: String, cb: Callback, loader: ClassLoader): Unit = {
    val clazz =
      try {
        loader.loadClass(className)
      } catch {
        case _: ClassNotFoundException       => return
        case _: NoClassDefFoundError         => return
        case _: ClassCastException           => return
        case _: UnsupportedClassVersionError => return
      }

    // Check if it is a subtype of `UnitTest` (and also not the definition of
    // `UnitTest` itself).
    if (clazz == classOf[UnitTest] || !classOf[UnitTest].isAssignableFrom(clazz))
      return

    // Check if this is a `BaseModule`, in which case we implicitly wrap its
    // constructor in a `Definition(...)` call.
    val isModule = classOf[BaseModule].isAssignableFrom(clazz)

    // Handle singleton objects by ensuring they are constructed.
    try {
      val field = clazz.getField("MODULE$")
      field.setAccessible(true)
      if (isModule)
        cb(className, () => Definition(field.get(null).asInstanceOf[BaseModule]))
      else
        cb(className, () => field.get(null))
      return
    } catch {
      case e: NoSuchFieldException => ()
    }

    // Handle classes by calling their constructor.
    try {
      if (isModule)
        cb(className, () => Definition(clazz.newInstance.asInstanceOf[BaseModule]))
      else
        cb(className, () => clazz.newInstance)
      return
    } catch {
      case e: NoSuchMethodException  => ()
      case e: IllegalAccessException => ()
    }
  }
}

/** A Chisel module that discovers and constructs all [[UnitTest]] subtypes
  * discovered in the classpath. This is just here as a convenience top-level
  * generator to collect all unit tests. In practice you would likely want to
  * use a command line utility that offers some additional filtering capability.
  */
class AllUnitTests extends RawModule {
  DiscoverUnitTests((_, gen) => gen(), Seq())
}
