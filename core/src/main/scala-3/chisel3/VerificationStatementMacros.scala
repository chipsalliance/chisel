// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal._

import scala.annotation.nowarn

object VerifStmtMacrosCompat {

  type SourceLineInfo = (String, Int)

  def formatFailureMessage(
    kind:     String,
    lineInfo: SourceLineInfo,
    cond:     Bool,
    message:  Option[Printable]
  )(
    using SourceInfo
  ): Printable = {
    val (filename, line) = lineInfo
    val lineMsg = s"$filename:$line".replaceAll("%", "%%")
    message match {
      case Some(msg) =>
        p"$kind failed: $msg\n"
      case None => p"$kind failed at $lineMsg\n"
    }
  }
}
