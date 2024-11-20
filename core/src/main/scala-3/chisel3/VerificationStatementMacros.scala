// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.PrintfMacrosCompat._
import chisel3.internal.firrtl.ir._
import chisel3.layer.block
import chisel3.layers
import chisel3.util.circt.IfElseFatalIntrinsic
import chisel3.internal.Builder.pushCommand
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
    implicit sourceInfo: SourceInfo
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
