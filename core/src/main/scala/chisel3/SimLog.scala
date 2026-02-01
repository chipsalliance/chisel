// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3._
import chisel3.internal.firrtl.ir.{Flush, Printf}
import chisel3.internal.Builder
import chisel3.experimental.SourceInfo

/** A file or I/O device to print to in simulation
  *
  * {{{
  * val log = SimLog.file("logfile.log")
  * log.printf(cf"in = $in%0d\n")
  * 
  * val stderr = SimLog.StdErr
  * stderr.printf(cf"in = $in%0d\n")
  * 
  * // SimLog filenames themselves can be Printable.
  * // Be careful to avoid uninitialized registers.
  * val idx = Wire(UInt(8.W))
  * val log2 = SimLog.file(cf"logfile_$idx%0d.log")
  * log2.printf(cf"in = $in%0d\n")
  * }}}
  */
sealed trait SimLog extends SimLogIntf {

  private[chisel3] def _printfImpl(pable: Printable)(implicit sourceInfo: SourceInfo): chisel3.printf.Printf = {
    this.printfWithReset(pable)(sourceInfo)
  }

  private[chisel3] def _flushImpl()(implicit sourceInfo: SourceInfo): Unit = {
    val clock = Builder.forcedClock
    _filename.foreach(Printable.checkScope(_, "SimLog filename "))

    when(!Module.reset.asBool) {
      layer.block(layers.Verification, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
        Builder.pushCommand(Flush(sourceInfo, _filename, clock.ref))
      }
    }
  }

  // Eventually this might be a richer type but for now Some[Printable] is filename, None is Stderr
  protected def _filename: Option[Printable]

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

    Printable.checkScope(pable, "printf ")
    _filename.foreach(Printable.checkScope(_, "SimLog filename "))

    layer.block(layers.Verification, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
      Builder.pushCommand(Printf(printfId, sourceInfo, _filename, clock.ref, pable))
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

// Uses firrtl fprintf with a Printable filename
private case class SimLogFile(filename: Printable) extends SimLog {
  override protected def _filename: Option[Printable] = Some(filename)
}

// Defaults to firrtl printf
private object StdErrSimLog extends SimLog {
  override protected def _filename: Option[Printable] = None
}

object SimLog extends SimLog$Intf {

  private[chisel3] def _fileImpl(filename: String)(implicit sourceInfo: SourceInfo): SimLog = {
    new SimLogFile(PString(filename))
  }

  private[chisel3] def _fileImpl(filename: Printable)(implicit sourceInfo: SourceInfo): SimLog = {
    new SimLogFile(filename)
  }

  /** The default FileDescriptor is stderr */
  val StdErr: SimLog = StdErrSimLog
}
