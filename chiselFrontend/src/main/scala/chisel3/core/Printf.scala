// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo

object printf { // scalastyle:ignore object.name
  /** Helper for packing escape characters */
  private[chisel3] def format(formatIn: String): String = {
    require(formatIn forall (c => c.toInt > 0 && c.toInt < 128),
      "format strings must comprise non-null ASCII values")
    def escaped(x: Char) = {
      require(x.toInt >= 0)
      if (x == '"' || x == '\\') {
        s"\\${x}"
      } else if (x == '\n') {
        "\\n"
      } else {
        require(x.toInt >= 32) // TODO \xNN once FIRRTL issue #59 is resolved
        x
      }
    }
    formatIn map escaped mkString ""
  }

  /** Prints a message in simulation.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), uses
    * whatever clock and reset are in scope.
    *
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def apply(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo): Unit =
    apply(Printable.pack(fmt, data:_*))
  /** Prints a message in simulation.
    *
    * Does not fire when in reset (defined as the encapsulating Module's
    * reset). If your definition of reset is not the encapsulating Module's
    * reset, you will need to gate this externally.
    *
    * May be called outside of a Module (like defined in a function), so
    * functions using printf make the standard Module assumptions (single clock
    * and single reset).
    *
    * @param pable [[Printable]] to print
    */
  def apply(pable: Printable)(implicit sourceInfo: SourceInfo): Unit = {
    when (!Builder.forcedReset) {
      printfWithoutReset(pable)
    }
  }

  private[chisel3] def printfWithoutReset(pable: Printable)(implicit sourceInfo: SourceInfo): Unit = {
    val clock = Builder.forcedClock
    pushCommand(Printf(sourceInfo, Node(clock), pable))
  }

  private[chisel3] def printfWithoutReset(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo): Unit =
    printfWithoutReset(Printable.pack(fmt, data:_*))
}
