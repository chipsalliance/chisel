package chisel3.experimental.util.algorithm

import chisel3._

/** Map each bits to logical or of itself and all bits less siginificant than it.
  * @example {{{
  * LSBOr("b00001000".U) // Returns "b11111000".U
  * LSBOr("b00010100".U) // Returns "b11111100".U
  * LSBOr("b00000000".U) // Returns "b00000000".U
  * }}}
  */
object LSBOr {
  def apply(data: UInt): UInt = VecInit(Seq.tabulate(data.getWidth) { i: Int =>
    VecInit(data.asBools().dropRight(data.getWidth - i - 1)).asUInt().orR()
  }).asUInt()
}

/** Map each bits to logical or of itself and all bits more siginificant than it.
  * @example {{{
  * MSBOr("b00001000".U) // Returns "b00001111".U
  * MSBOr("b00010100".U) // Returns "b00011111".U
  * MSBOr("b00000000".U) // Returns "b00000000".U
  * }}}
  */
object MSBOr {
  def apply(data: UInt): UInt = VecInit(Seq.tabulate(data.getWidth) { i: Int =>
    VecInit(data.asBools().drop(i)).asUInt().orR()
  }).asUInt()
}
