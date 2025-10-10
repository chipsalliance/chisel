// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.testing.HasTestingDirectory
import java.nio.file.Files

/** Type class for providing a simulator. */
trait HasSimulator {

  /** Return a simulator.  This require providing a type class implementation of
    * [[HasTestingDirectory]].
    */
  def getSimulator(implicit testingDirectory: HasTestingDirectory): Simulator[_]

}

/** Type class implementations of [[HasSimulator]]. */
object HasSimulator {

  /** This object provides implementations of [[HasSimulator]].  To set the
    * simulator in your test, please import one of the implementations.
    *
    * E.g., to use Verilator import:
    *
    * {{{
    * import chisel3.simulator.HasSimulator.simulators.verilator
    * }}}
    *
    * Or, to use VCS import:
    *
    * {{{
    * import chisel3.simulator.HasSimulator.simulators.vcs
    * }}}
    *
    * Note: if you do _not_ import one of these, the default will be to use
    * Verilator due to the low-priority implicit default
    * [[HasSimulator.default]].
    */
  object simulators {

    /** A [[HasSimulator]] implementation for a Verilator simulator. */
    def verilator(
      compilationSettings: svsim.CommonCompilationSettings = svsim.CommonCompilationSettings(),
      verilatorSettings: svsim.verilator.Backend.CompilationSettings =
        svsim.verilator.Backend.CompilationSettings.default
    ): HasSimulator = new HasSimulator {
      override def getSimulator(implicit testingDirectory: HasTestingDirectory): Simulator[svsim.verilator.Backend] =
        new Simulator[svsim.verilator.Backend] {
          override val backend = svsim.verilator.Backend.initializeFromProcessEnvironment()
          override val tag = "verilator"
          override val commonCompilationSettings = compilationSettings
          override val backendSpecificCompilationSettings = verilatorSettings
          override val workspacePath = Files.createDirectories(testingDirectory.getDirectory).toString
        }
    }

    /** A [[HasSimulator]] implementation for a VCS simulator. */
    def vcs(
      compilationSettings: svsim.CommonCompilationSettings = svsim.CommonCompilationSettings(),
      vcsSettings:         svsim.vcs.Backend.CompilationSettings = svsim.vcs.Backend.CompilationSettings()
    ): HasSimulator = new HasSimulator {
      override def getSimulator(implicit testingDirectory: HasTestingDirectory): Simulator[svsim.vcs.Backend] =
        new Simulator[svsim.vcs.Backend] {
          override val backend = svsim.vcs.Backend.initializeFromProcessEnvironment()
          override val tag = "vcs"
          override val commonCompilationSettings = compilationSettings
          override val backendSpecificCompilationSettings = vcsSettings
          override val workspacePath = Files.createDirectories(testingDirectory.getDirectory).toString
        }
    }

  }

  /** Low-priority default implementation of [[HasSimulator]] that uses Verilator.
    * This is the default that will be used if the user does provide an
    * alternative.
    */
  implicit def default: HasSimulator = simulators.verilator()

}
