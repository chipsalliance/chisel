// See LICENSE for license details.

/** LFSRs in all shapes and sizes.
  */

package chisel3.util

import chisel3._
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order
import chisel3.util.random.FibonacciLFSR

/** LFSR16 generates a 16-bit linear feedback shift register, returning the register contents.
  * This is useful for generating a pseudo-random sequence.
  *
  * The example below, taken from the unit tests, creates two 4-sided dice using `LFSR16` primitives:
  * @example {{{
  *   val bins = Reg(Vec(8, UInt(32.W)))
  *
  *   // Create two 4 sided dice and roll them each cycle.
  *   // Use tap points on each LFSR so values are more independent
  *   val die0 = Cat(Seq.tabulate(2) { i => LFSR16()(i) })
  *   val die1 = Cat(Seq.tabulate(2) { i => LFSR16()(i + 2) })
  *
  *   val rollValue = die0 +& die1  // Note +& is critical because sum will need an extra bit.
  *
  *   bins(rollValue) := bins(rollValue) + 1.U
  *
  * }}}
  */
// scalastyle:off magic.number
@deprecated("LFSR16 is deprecated in favor of the parameterized chisel3.util.random.LFSR", "3.2")
object LFSR16 {
  /** Generates a 16-bit linear feedback shift register, returning the register contents.
    * @param increment optional control to gate when the LFSR updates.
    */
  @deprecated("Use chisel3.util.random.LFSR(16) for a 16-bit LFSR", "3.2")
  @chiselName
  def apply(increment: Bool = true.B): UInt =
    Vec( FibonacciLFSR
          .maxPeriod(16, increment, seed = Some(BigInt(1) << 15))
          .asBools
          .reverse )
      .asUInt

}
// scalastyle:on magic.number
