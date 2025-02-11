package chisel3.simulator

import java.io.File
import java.nio.file.{FileSystems, Files, Path}
import java.time.LocalDateTime
import scala.reflect.io.Directory

/** This is a trait that can be mixed into a class to determine where
  * compilation should happen and where simulation artifacts should be written.
  */
trait HasTestingDirectory {

  /** Return the directory where tests should be placed. */
  def getDirectory(testClassName: String): Path

}

/** This provides some default implementations of the [[HasTestingDirectory]]
  * type class.
  */
object HasTestingDirectory {

  /** An implementation of [[HasTestingDirectory]] which will use the class name
    * timestamp.  When this implementation is used, everything is put in a
    * `test_run_dir/<class-name>/<timestamp>/`.  E.g., this may produce
    * something like:
    *
    * {{{
    * test_run_dir/
    * └── DefaultSimulator
    *     ├── 2025-02-05T16-58-02.175175
    *     ├── 2025-02-05T16-58-11.941263
    *     └── 2025-02-05T16-58-17.721776
    * }}}
    */
  val timestamp: HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory(testClassName: String) = FileSystems
      .getDefault()
      .getPath("test_run_dir/", testClassName, LocalDateTime.now().toString().replace(':', '-'))
  }

  /** An implementation generator of [[HasTestingDirectory]] which will use an
    * operating system-specific temporary directory.  This file can optionally
    * be deleted when the JVM shuts down.
    *
    * @param deleteOnExit if true, delete the temporary directory when the JVM exits
    */
  def temporary(deleteOnExit: Boolean = true): HasTestingDirectory = new HasTestingDirectory {
    override def getDirectory(testClassName: String) = {
      val path = FileSystems
        .getDefault()
        .getPath(
          Files.createTempDirectory(s"${testClassName}-${LocalDateTime.now().toString().replace(':', '-')}").toString
        )
      if (deleteOnExit) {
        sys.addShutdownHook(new Directory(path.toFile).deleteRecursively())
      }
      path
    }
  }

  /** The default testing directory behavior.  If the user does _not_ provide an
    * alternative type class implementation of [[HasTestingDirectory]], then
    * this will be what is used.
    */
  implicit val default: HasTestingDirectory = timestamp

}
