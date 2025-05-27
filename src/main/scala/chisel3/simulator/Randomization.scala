// SPDX-License-Identifier: Apache-2.0

package chisel3.simulator

import svsim.CommonCompilationSettings.VerilogPreprocessorDefine

/** A description of how a Chisel circuit should be randomized
  *
  * @param registers if true, randomize the initial state of registers
  * @param memories if true, randomize the initial state of memories
  * @param garbageAssign
  * @param invalidAssign
  * @param delay an optional delay value to apply to the randomization.  This
  * will cause the randomization to be applied this many Verilog time units
  * after the simulation starts.
  * @throws IllegalArgumentException if register and memory randomization are
  * both disabled and delay or randomValue are non-empty
  */
final class Randomization(
  val registers:   Boolean,
  val memories:    Boolean,
  val delay:       Option[Int],
  val randomValue: Option[String]
) {

  require(
    registers || memories || (delay.isEmpty && randomValue.isEmpty),
    "when register and memory randomization is disabled, then `delay` and `randomValue` should be empty as they have no effect"
  )

  require(
    delay.isEmpty || delay.get != 0,
    "a delay of zero is illegal as this can conflict with initial blocks and simulator-specific time-zero behavior"
  )

  /** Create a copy of this [[Randomization]] changing some parameters */
  def copy(
    registers:   Boolean = registers,
    memories:    Boolean = memories,
    delay:       Option[Int] = delay,
    randomValue: Option[String] = randomValue
  ) = new Randomization(registers, memories, delay, randomValue)

  /** Convert this to Verilog preprocessor defines */
  private[simulator] def toPreprocessorDefines: Seq[VerilogPreprocessorDefine] = {
    Option.when(registers)(VerilogPreprocessorDefine("RANDOMIZE_REG_INIT")).toSeq ++
      Option.when(memories)(VerilogPreprocessorDefine("RANDOMIZE_MEM_INIT")) ++
      delay.map(d => VerilogPreprocessorDefine("RANDOMIZE_DELAY", d.toString)) ++
      randomValue.map(VerilogPreprocessorDefine("RANDOM", _))
  }

}

object Randomization {

  /** Randomize nothing
    *
    * This will cause the simulation to be brought up in whatever state the
    * simulator brings up a simulation in.  If the simulator supports `x`, then
    * uninitialized hardware will be brought up in `x`.  However, if the
    * simulator is two-state (e.g., Verilator), then it will be brought up in a
    * simulator-dependent state.
    */
  def uninitialized = new Randomization(
    registers = false,
    memories = false,
    delay = None,
    randomValue = None
  )

  /** Randomize everything
    *
    * This will randomize everything that the FIRRTL/Verilog ABI allows.  All
    * Verilog that Chisel produces will have a random two-state value.  Verilog
    * that Chisel does not have control of (e.g., blackboxes) will be brought up
    * in a different state unless they opt-in to the FIRRTL/Verilog ABI.
    *
    * Non-two-state values (i.e., `x`)
    *
    * @note The FIRRTL/Verilog ABI for randomization is undocumented in the
    * FIRRTL ABI specification.
    */
  def random = new Randomization(
    registers = true,
    memories = true,
    delay = Some(1),
    randomValue = Some("$urandom")
  )

}
