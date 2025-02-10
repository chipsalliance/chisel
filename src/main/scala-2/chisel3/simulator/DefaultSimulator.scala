// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import svsim._
import chisel3.{Module, RawModule}
import chisel3.util.simpleClassName
import java.nio.file.Files

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
