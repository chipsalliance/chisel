package chisel3.simulator

import java.io.IOException
import java.nio.file._
import java.nio.file.attribute.BasicFileAttributes
import java.time.LocalDateTime
import java.util.Comparator

/** This is a trait that can be mixed into a class to determine where
  * compilation should happen and where simulation artifacts should be written.
  */
trait HasTestingDirectory {

  /** Return the directory where tests should be placed. */
  def getDirectory: Path

}

/** This provides some default implementations of the [[HasTestingDirectory]]
  * type class.
  */
object HasTestingDirectory {

  /** An implementation of [[HasTestingDirectory]] which will use the
    * class name timestamp.  When this implementation is used,
    * everything is put in a `test_run_dir/<class-name>/<timestamp>/`.
    * E.g., this may produce something like:
    *
    * {{{
    * test_run_dir
    * └── chiselsim
    *     ├── 2025-02-05T16-58-02.175175
    *     ├── 2025-02-05T16-58-11.941263
    *     └── 2025-02-05T16-58-17.721776
    * }}}
    */
  val timestamp: HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory: Path = FileSystems
      .getDefault()
      .getPath("test_run_dir", "chiselsim", LocalDateTime.now().toString.replace(':', '-'))
  }

  /** An implementation generator of [[HasTestingDirectory]] which will use an
    * operating system-specific temporary directory. This directory can
    * optionally be deleted when the JVM shuts down.
    *
    * @param deleteOnExit if true, delete the temporary directory when the JVM exits
    */
  def temporary(deleteOnExit: Boolean = true): HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory: Path = {
      val dir = Files.createTempDirectory(
        s"chiselsim-${LocalDateTime.now().toString.replace(':', '-')}"
      )

      if (deleteOnExit) {
        sys.addShutdownHook {
          deleteDir(dir)
        }
      }
      dir
    }
  }

  /** The default testing directory behavior. If the user does _not_ provide an
    * alternative type class implementation of [[HasTestingDirectory]], then
    * this will be what is used.
    */
  implicit val default: HasTestingDirectory = timestamp

  /** Walk directory and delete contents */
  private def deleteDir(directory: Path): Unit = {
    if (Files.exists(directory)) {
      Files
        .walk(directory)
        .sorted(Comparator.reverseOrder())
        .forEach(Files.delete)
    }
  }
}
