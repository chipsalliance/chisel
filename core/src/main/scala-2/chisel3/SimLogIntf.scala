// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.PrintfMacrosCompat
import chisel3.experimental.SourceInfo

import scala.language.experimental.macros

private[chisel3] trait SimLogIntf { self: SimLog =>

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
    * ==Format Strings==
    *
    * This method expects a ''format string'' and an ''argument list'' in a similar style to printf
    * in C. The format string expects a [[scala.Predef.String String]] that may contain ''format
    * specifiers'' For example:
    * {{{
    *   printf("myWire has the value %d\n", myWire)
    * }}}
    * This prints the string "myWire has the value " followed by the current value of `myWire` (in
    * decimal, followed by a newline.
    *
    * There must be exactly as many arguments as there are format specifiers
    *
    * ===Format Specifiers===
    *
    * Format specifiers are prefixed by `%`. If you wish to print a literal `%`, use `%%`.
    *   - `%d` - Decimal
    *   - `%x` - Hexadecimal
    *   - `%b` - Binary
    *   - `%c` - 8-bit Character
    *   - `%n` - Name of a signal
    *   - `%N` - Full name of a leaf signal (in an aggregate)
    *   - `%%` - Literal percent
    *   - `%m` - Hierarchical name of the current module
    *   - `%T` - Simulation time
    *
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def printf(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf =
    macro PrintfMacrosCompat._applyMacroWithInterpolatorCheck

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
  def printf(pable: Printable)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf =
    _printfImpl(pable)

  /** Flush any buffered output immediately */
  def flush()(implicit sourceInfo: SourceInfo): Unit =
    _flushImpl()
}

private[chisel3] trait SimLog$Intf { self: SimLog.type =>

  /** Print to a file given by `filename`
    */
  def file(filename: String)(implicit sourceInfo: SourceInfo): SimLog =
    _fileImpl(filename)

  def file(filename: Printable)(implicit sourceInfo: SourceInfo): SimLog =
    _fileImpl(filename)
}
