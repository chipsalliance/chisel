// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo

/** Prints a message in simulation
  *
  * See apply methods for use
  */
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
    *
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def apply(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =
    apply(Printable.pack(fmt, data:_*))
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
  def apply(pable: Printable)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    when (!Module.reset.toBool) {
      printfWithoutReset(pable)
    }
  }

  private[chisel3] def printfWithoutReset(pable: Printable)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit = {
    val clock = Builder.forcedClock
    pushCommand(Printf(sourceInfo, Node(clock), pable))
  }
  private[chisel3] def printfWithoutReset(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Unit =
    printfWithoutReset(Printable.pack(fmt, data:_*))
}
