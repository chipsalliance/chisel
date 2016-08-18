// See LICENSE for license details.

package chisel3.core

import scala.language.experimental.macros

import chisel3.internal._
import chisel3.internal.Builder.pushCommand
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.SourceInfo

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
    when (!Builder.forcedModule.reset) {
      printfWithoutReset(fmt, data:_*)
    }
  }

  private[core] def printfWithoutReset(fmt: String, data: Bits*)(implicit sourceInfo: SourceInfo) {
    val clock = Builder.forcedModule.clock
    pushCommand(Printf(sourceInfo, Node(clock), fmt, data.map((d: Bits) => d.ref)))
  }
}
