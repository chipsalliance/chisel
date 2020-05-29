/*
rule = ChiselLinter
*/
package fix

import chisel3._

object ChiselLinter {
  123.U(6) // assert: ChiselLinter
  val x = 3.U
  val y = 3.U
  y == x // assert: ChiselLinter
}
