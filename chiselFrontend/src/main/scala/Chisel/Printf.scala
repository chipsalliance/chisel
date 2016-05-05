// See LICENSE for license details.

package Chisel

import scala.language.experimental.macros

import internal._
import internal.Builder.pushCommand
import internal.firrtl._
import internal.sourceinfo.SourceInfo

object printf { // scalastyle:ignore object.name
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
    * @param fmt printf format string
    * @param data format string varargs containing data to print
    */
  def apply(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo) {
    when (!(Builder.dynamicContext.currentModule.get.reset)) {
      printfWithoutReset(fmt, data:_*)
    }
  }

  private[Chisel] def printfWithoutReset(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo) {
    val clock = Builder.dynamicContext.currentModule.get.clock
    pushCommand(Printf(sourceInfo, Node(clock), fmt, data.map((d: Bits) => d.ref)))
  }
}
