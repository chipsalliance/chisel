// See LICENSE for license details.

/** LFSRs in all shapes and sizes.
  */

package chisel3.util

import chisel3._
import chisel3.internal.naming.chiselName  // can't use chisel3_ version because of compile order
//import chisel3.core.ExplicitCompileOptions.Strict

// scalastyle:off magic.number
object LFSR16 {
  /** Generates a 16-bit linear feedback shift register, returning the register contents.
    * May be useful for generating a pseudorandom sequence.
    *
    * @param increment optional control to gate when the LFSR updates.
    */
  @chiselName
  def apply(increment: Bool = true.B): UInt = {
    val width = 16
    val lfsr = RegInit(1.U(width.W))
    when (increment) { lfsr := Cat(lfsr(0)^lfsr(2)^lfsr(3)^lfsr(5), lfsr(width-1,1)) }
    lfsr
  }
}
// scalastyle:on magic.number

