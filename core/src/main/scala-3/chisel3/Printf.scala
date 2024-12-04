// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.experimental.SourceInfo
import chisel3.PrintfMacrosCompat._

import scala.language.experimental.macros

/** Prints a message in simulation
  *
  * See apply methods for use
  */
object printf {

  /** Named class for [[printf]]s. */
  final class Printf private[chisel3] (val pable: Printable) extends VerificationStatement

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

  /** Prints a message in simulation
    *
    * Prints a message every cycle. If defined within the scope of a [[when]] block, the message
    * will only be printed on cycles that the when condition is true.
    *
    * Does not fire when in reset (defined as the encapsulating Module's reset). If your definition
    * of reset is not the encapsulating Module's reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), uses the current default clock
    * and reset. These can be overriden with [[withClockAndReset]].
    *
    * @see [[Printable]] documentation
    * @param pable [[Printable]] to print
    */
  def apply(pable: Printable)(using sourceInfo: SourceInfo): chisel3.printf.Printf =
    PrintfMacrosCompat.printfWithReset(pable)(using sourceInfo)
}
