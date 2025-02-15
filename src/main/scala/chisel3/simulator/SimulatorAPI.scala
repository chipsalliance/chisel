// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import chisel3.{Module, RawModule}
import chisel3.util.simpleClassName
import java.nio.file.Files

trait SimulatorAPI {

  /** Simulate a [[RawModule]] without any initialization procedure.
    *
    * Use of this method is not advised when [[simulate]] can be used instead.
    * This method may cause problems with certain simulators as it is up to the
    * user to ensure that race conditions will not exist during the time zero
    * and reset procedure.
    *
    * @param module the Chisel module to generate
    * @param layerControl layers that should be enabled
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    */
  def simulateRaw[T <: RawModule](
    module:         => T,
    chiselSettings: ChiselSettings[T] = ChiselSettings.defaultRaw[T]
  )(stimulus: (T) => Unit)(implicit hasSimulator: HasSimulator, testingDirectory: HasTestingDirectory): Unit = {

    hasSimulator.getSimulator
      .simulate(module, chiselSettings) { module =>
        stimulus(module.wrapped)
      }
      .result
  }

  /** Simulate a [[Module]] using a standard initialization procedure that is
    * suitable for any Chisel-generated Verilog module.  The commands specified
    * in the `body` will run _after_ this initialization procedure.
    *
    * The initialization procedure is as follows:
    *
    *     time 0:     bring everything up using simulator settings
    *     time 1:     bring reset out of `x` and deassert it.
    *     time 2:     assert reset
    *     time 3:     first clock edge
    *     time 4 + n: deassert reset (where n == `additionalResetCycles`)
    *
    * This is doing several times:
    *
    *   1. There is guaranteed to be a time when FIRRTL/Verilog-based
    *      randomization can happen at _either_ time 0 or time 1.)
    *   2. If time 1 is used for FIRRTL/Verilog-based randomization, then time 0
    *      can be used for simulator-based initialization, e.g.,
    *      `+vcs+initreg+random`.  Simulator initialization will race with
    *      FIRRTL/Verilog-based randomization and it is critical that they do
    *      not happen at the same time.
    *   3. Both FIRRTL/Verilog-based randomization and simulator-based
    *      randomization should not occur on a clock edge, e.g., an edge when
    *      reset is asserted.  This can be yet-another race condition that has
    *      to be avoided.
    *   4. Reset always sees a posedge.  This avoids problems with asynchronous
    *      reset logic behavior where they may (correctly in Verilog) _not_ fire
    *      if you bring the design with reset asserted.  Note: it would be fine
    *      to do an `x -> 1` transition to show an edge, however, it looks
    *      cleaner to bring reset to `0`.
    *
    * @param module the Chisel module to generate
    * @param layerControl layers that should be enabled
    * @param additionalResetCycles a number of _additional_ cycles to assert
    * reset for
    * @param stimulus directed stimulus to use
    * @param testingDirectory a type class implementation that can be used to
    * change the behavior of where files will be created
    */
  def simulate[T <: Module](
    module:                => T,
    chiselSettings:        ChiselSettings[T] = ChiselSettings.default[T],
    additionalResetCycles: Int = 0
  )(stimulus: (T) => Unit)(implicit hasSimulator: HasSimulator, testingDirectory: HasTestingDirectory): Unit = {

    hasSimulator.getSimulator
      .simulate(module, chiselSettings) { module =>
        val dut = module.wrapped
        val reset = module.port(dut.reset)
        val clock = module.port(dut.clock)
        val controller = module.controller

        // Run the initialization procedure.
        controller.run(1)
        reset.set(0)
        controller.run(1)
        reset.set(1)
        clock.tick(
          timestepsPerPhase = 1,
          maxCycles = 1 + additionalResetCycles,
          inPhaseValue = 0,
          outOfPhaseValue = 1,
          sentinel = None
        )
        reset.set(0)
        controller.run(0)

        // Run the user code.
        stimulus(dut)
      }
      .result

  }

}
