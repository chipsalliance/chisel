// See LICENSE for license details.

package chisel3.util.random

import chisel3._
import chisel3.util.Valid

/** Pseudo Random Number Generators (PRNG) interface
  * @param n the width of the LFSR
  */
class PRNGIO(val n: Int) extends Bundle {

  /** A [[chisel3.util.Valid Valid]] interface that can be used to set the seed */
  val seed = Input(Valid(UInt(n.W)))

  /** When asserted, the PRNG will increment by one */
  val increment = Input(Bool())

  /** The current state of the PRNG */
  val out = Output(UInt(n.W))
}

/** An abstract class representing a Pseudo Random Number Generator (PRNG)
  * @param width the width of the PRNG
  * @param seed the initial state of the PRNG
  * @param step the number of state updates per cycle
  * @param updateSeed if true, when loading the seed the state will be updated as if the seed were the current state, if
  * false, the state will be set to the seed
  */
abstract class PRNG(val width: Int, val seed: Option[BigInt], step: Int = 1, updateSeed: Boolean = false) extends Module {
  require(width > 0, s"Width must be greater than zero! (Found '$width')")
  require(step > 0, s"Step size must be greater than one! (Found '$step')")

  val io: PRNGIO = IO(new PRNGIO(width))

  /** Internal state of the PRNG. If the user sets a seed, this is initialized to the seed. If the user does not set a
    * seed this is left uninitialized. In the latter case, a PRNG subclass *must do something to handle lockup*, e.g.,
    * the PRNG state should be manually reset to a safe value. E.g., [[LFSR]] will, based on the chosen reduction
    * operator, either set or reset the least significant bit of the state.
    */
  val state: UInt = seed match {
    case Some(s) => RegInit(s.U(width.W))
    case None    => Reg(UInt(width.W))
  }

  /** State update function
    * @param s input state
    * @return the next state
    */
  def delta(s: UInt): UInt

  /** The method that will be used to update the state of this PRNG
    * @param s input state
    * @return the next state after `step` applications of [[PRNG.delta]]
    */
  final def nextState(s: UInt): UInt = (0 until step).foldLeft(s){ case (s, _) => delta(s) }

  when (io.increment) {
    state := nextState(state)
  }

  when (io.seed.fire()) {
    state := (if (updateSeed) { nextState(io.seed.bits) } else { io.seed.bits })
  }

  io.out := state

}

/** Helper utilities related to the construction of Pseudo Random Number Generators (PRNGs) */
object PRNG {

  /** Wrap a [[PRNG]] to only return a pseudo-random [[UInt]]
    * @param gen a pseudo random number generator
    * @param increment when asserted the [[PRNG]] will increment
    * @return the output (internal state) of the [[PRNG]]
    */
  def apply(gen: => PRNG, increment: Bool = true.B): UInt = {
    val prng = Module(gen)
    prng.io.seed.valid := false.B
    prng.io.seed.bits := DontCare
    prng.io.increment := increment
    prng.io.out
  }

}
