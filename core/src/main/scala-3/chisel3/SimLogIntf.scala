// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait SimLogIntf { self: SimLog =>

  /** Prints a message in simulation
    *
    * Prints a message every cycle. If defined within the scope of a
    * [[when]] block, the message will only be printed on cycles that
    * the when condition is true.
    *
    * Does not fire when in reset (defined as the encapsulating
    * Module's reset). If your definition of reset is not the
    * encapsulating Module's reset, you will need to gate this
    * externally.
    *
    * May be called outside of a Module (like defined in a function),
    * uses the current default clock and reset. These can be overriden
    * with [[withClockAndReset]].
    *
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def printf(fmt: String, data: Bits*)(using sourceInfo: SourceInfo): chisel3.printf.Printf =
    self.printf(Printable.pack(fmt, data*))
}
