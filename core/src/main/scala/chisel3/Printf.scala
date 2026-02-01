// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

/** Prints a message in simulation
  *
  * See apply methods for use
  */
object printf extends PrintfIntf {

  /** Named class for [[printf]]s. */
  final class Printf private[chisel3] (val pable: Printable) extends VerificationStatement

  private[chisel3] def _applyPrintableImpl(pable: Printable)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf =
    SimLog.StdErr._printfImpl(pable)(sourceInfo)

  /** Helper for packing escape characters */
  private[chisel3] def format(formatIn: String): String = {
    require(formatIn.forall(c => c.toInt > 0 && c.toInt < 128), "format strings must comprise non-null ASCII values")
    def escaped(x: Char) = {
      require(x.toInt >= 0, s"char ${x} to Int ${x.toInt} must be >= 0")
      if (x == '"' || x == '\\') {
        s"\\${x}"
      } else if (x == '\n') {
        "\\n"
      } else if (x == '\t') {
        "\\t"
      } else {
        require(
          x.toInt >= 32,
          s"char ${x} to Int ${x.toInt} must be >= 32"
        ) // TODO \xNN once FIRRTL issue #59 is resolved
        x
      }
    }
    formatIn.map(escaped).mkString("")
  }
}
