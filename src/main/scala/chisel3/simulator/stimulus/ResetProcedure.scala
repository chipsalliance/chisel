// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator.stimulus

import chisel3.{Clock, Module, RawModule, Reset}
import chisel3.simulator.AnySimulatedModule

/** Stimulus that will apply a standard reset procedure to a Chisel circuit.
  *
  * The reset procedure is as follows:
  *
  *     time 0:     bring everything up using simulator settings
  *     time 1:     bring reset out of `x` and deassert it.
  *     time 2:     assert reset
  *     time 3:     first clock edge
  *     time 4 + n: deassert reset (where n == `additionalResetCycles`)
  *
  * This intentionally structured to guarantee the following properties:
  *
  *   1. There is guaranteed to be a time when FIRRTL/Verilog-based
  *      randomization can happen at _either_ time 0 or time 1.)
  *   2. If time 1 is used for FIRRTL/Verilog-based randomization, then time 0
  *      can be used for simulator-based initialization, e.g.,
  *      `+vcs+initreg+random`.  Simulator initialization will race with
  *      FIRRTL/Verilog-based randomization and it is critical that they do not
  *      happen at the same time.
  *   3. Both FIRRTL/Verilog-based randomization and simulator-based
  *      randomization should not occur on a clock edge, e.g., an edge when
  *      reset is asserted.  This can be yet-another race condition that has to
  *      be avoided.
  *   4. Reset always sees a posedge.  This avoids problems with asynchronous
  *      reset logic behavior where they may (correctly in Verilog) _not_ fire
  *      if you bring the design with reset asserted.  Note: it would be fine to
  *      do an `x -> 1` transition to show an edge, however, it looks cleaner to
  *      bring reset to `0`.
  */
trait ResetProcedure[A] extends Stimulus.Type[A] {

  /** The number of additional reset cycles to hold the reset for. */
  protected def _additionalResetCycles: Int

  /** A function to clock to use. */
  protected def _getClock: (A) => Clock

  /** A function that returns the reset to use. */
  protected def _getReset: (A) => Reset

  /** Apply reset procedure stimulus. */
  override final def apply(dut: A): Unit = {

    val module = AnySimulatedModule.current
    val controller = module.controller

    val reset = module.port(_getReset(dut))
    val clock = module.port(_getClock(dut))

    // Run the initialization procedure.
    controller.run(1)
    reset.set(0)
    controller.run(1)
    reset.set(1)
    clock.tick(
      timestepsPerPhase = 1,
      maxCycles = 1 + _additionalResetCycles,
      inPhaseValue = 0,
      outOfPhaseValue = 1,
      sentinel = None
    )
    reset.set(0)
    controller.run(0)

  }

}

/** Factory of [[ResetProcedure]] stimulus. */
object ResetProcedure {

  /** Return reset stimulus for a [[Module]]. */
  def module[A <: Module](additionalResetCycles: Int = 0): ResetProcedure[A] = new ResetProcedure[A] {

    override protected final val _additionalResetCycles = additionalResetCycles

    override protected final val _getClock = _.clock

    override protected final val _getReset = _.reset

  }

  /** Return reset stimulus for a [[RawModule]].
    *
    * This necessarily requires defining how to get the clock and reset to use
    * to apply the reset.
    *
    * @param getClock return the clock to use
    * @param getReset return the reset to use
    * @param additionalResetCycles the reset to use
    */
  def any[A <: RawModule](
    getClock:              A => Clock,
    getReset:              A => Reset,
    additionalResetCycles: Int = 0
  ): ResetProcedure[A] = new ResetProcedure[A] {

    override protected final val _additionalResetCycles = additionalResetCycles

    override protected final val _getClock = getClock

    override protected final val _getReset = getReset

  }

}
