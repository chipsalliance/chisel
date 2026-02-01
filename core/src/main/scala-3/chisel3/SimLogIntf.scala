// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo

private[chisel3] trait SimLogIntf { self: SimLog =>
  // TODO implement macro for Scala 3 for printf(fmt: String, data: Bits*)

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
  def printf(pable: Printable)(using sourceInfo: SourceInfo): chisel3.printf.Printf =
    _printfImpl(pable)

  /** Flush any buffered output immediately */
  def flush()(using sourceInfo: SourceInfo): Unit =
    _flushImpl()
}

private[chisel3] trait SimLog$Intf { self: SimLog.type =>

  /** Print to a file given by `filename`
    */
  def file(filename: String)(using sourceInfo: SourceInfo): SimLog =
    _fileImpl(filename)

  def file(filename: Printable)(using sourceInfo: SourceInfo): SimLog =
    _fileImpl(filename)
}
