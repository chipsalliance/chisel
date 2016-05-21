// See LICENSE for license details.

/** LFSRs in all shapes and sizes.
  */

package chisel

// scalastyle:off magic.number
/** linear feedback shift register
  */
object LFSR16
{
  def apply(increment: Bool = Bool(true)): UInt =
  {
    val width = 16
    val lfsr = Reg(init=UInt(1, width))
    when (increment) { lfsr := Cat(lfsr(0)^lfsr(2)^lfsr(3)^lfsr(5), lfsr(width-1,1)) }
    lfsr
  }
}
// scalastyle:on magic.number

