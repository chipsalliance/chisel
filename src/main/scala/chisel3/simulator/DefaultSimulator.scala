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
object DefaultSimulator extends PeekPokeAPI with SimulatorAPI
