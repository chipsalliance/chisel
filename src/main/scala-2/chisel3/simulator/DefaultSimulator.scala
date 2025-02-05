package chisel3.simulator

import svsim._
import chisel3.RawModule
import chisel3.util.simpleClassName
import java.nio.file.{FileSystems, Files, Path}
import java.time.LocalDateTime

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

  /** The default testing directory behavior.  If the user does _not_ provide an
    * alternative type class implementation of [[HasTestingDirectory]], then
    * this will be what is used.
    */
  implicit val default: HasTestingDirectory = timestamp

}

/** Provides a simple API for running simulations.  This will write temporary
  * outputs to a directory derived from the class name of the test.
  *
  * @example
  * {{{
  * import chisel3.simulator.DefaultSimulator._
  * ...
  * simulate(new MyChiselModule()) { module => ... }
  * }}}
  */
object DefaultSimulator extends PeekPokeAPI {

  private class DefaultSimulator(val workspacePath: String) extends SingleBackendSimulator[verilator.Backend] {
    val backend = verilator.Backend.initializeFromProcessEnvironment()
    val tag = "default"
    val commonCompilationSettings = CommonCompilationSettings()
    val backendSpecificCompilationSettings = verilator.Backend.CompilationSettings()
  }

  def simulate[T <: RawModule](
    module:       => T,
    layerControl: LayerControl.Type = LayerControl.EnableAll
  )(body: (T) => Unit)(implicit testingDirectory: HasTestingDirectory): Unit = {

    val testClassName = simpleClassName(getClass())

    val simulator = new DefaultSimulator(
      workspacePath = Files.createDirectories(testingDirectory.getDirectory(testClassName)).toString
    )

    simulator.simulate(module, layerControl)({ module => body(module.wrapped) }).result
  }

}
