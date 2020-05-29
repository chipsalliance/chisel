/*
rule = ChiselLinter
*/
package fix

import chisel3._

object ChiselLinter {
  123.U(6) // assert: ChiselLinter
}
