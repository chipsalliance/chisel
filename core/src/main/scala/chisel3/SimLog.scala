// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal.firrtl.ir.Printf
import chisel3.internal.Builder
import chisel3.experimental.SourceInfo

/** A file or I/O device to print to in simulation
  * 
  * {{{
  * val fd = FileDescriptor("logfile.log")
  * fd.printf(cf"in = $in%0d\n")
  * }}}
  */
sealed trait SimLog extends SimLogIntf {

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
  def printf(pable: Printable)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf = {
    this.printfWithReset(pable)(sourceInfo)
  }

  // Eventually this might be a richer type but for now Some[String] is filename, None is Stderr
  protected def _filename: Option[String]

  private[chisel3] def printfWithReset(
    pable: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): chisel3.printf.Printf = {
    var printfId: chisel3.printf.Printf = null
    when(!Module.reset.asBool) {
      printfId = printfWithoutReset(pable)
    }
    printfId
  }

  private[chisel3] def printfWithoutReset(
    pable: Printable
  )(
    implicit sourceInfo: SourceInfo
  ): chisel3.printf.Printf = {
    val clock = Builder.forcedClock
    val printfId = new chisel3.printf.Printf(pable)

    Printable.checkScope(pable)

    layer.block(layers.Verification, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
      Builder.pushCommand(Printf(printfId, sourceInfo, this._filename, clock.ref, pable))
    }
    printfId
  }

  private[chisel3] def printfWithoutReset(
    fmt:  String,
    data: Bits*
  )(
    implicit sourceInfo: SourceInfo
  ): chisel3.printf.Printf =
    printfWithoutReset(Printable.pack(fmt, data: _*))
}

// Uses firrtl fprintf with a String filename
private case class SimLogFile(filename: String) extends SimLog {
  override protected def _filename: Option[String] = Some(filename)
}

// Defaults to firrtl printf
private object StdErrSimLog extends SimLog {
  override protected def _filename: Option[String] = None
}

object SimLog {

  /** Print to a file given by `filename`
    */
  def file(filename: String)(implicit sourceInfo: SourceInfo): SimLog = {
    new SimLogFile(filename)
  }

  /** The default FileDescriptor is stderr */
  val StdErr: SimLog = StdErrSimLog
}
