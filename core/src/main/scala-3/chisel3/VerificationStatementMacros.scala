// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.internal._
import chisel3.internal.firrtl.ir._
import chisel3.util.circt.IfElseFatalIntrinsic
import chisel3.layer.block

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

  object assert {
    private def emitIfElseFatalIntrinsic(
      clock:     Clock,
      predicate: Bool,
      enable:    Bool,
      format:    Printable
    )(
      implicit sourceInfo: SourceInfo
    ): chisel3.assert.Assert = {
      block(layers.Verification.Assert, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
        val id = Builder.forcedUserModule // It should be safe since we push commands anyway.
        val args = Printable.unpackFirrtlArgs(format)
        IfElseFatalIntrinsic(id, format, "chisel3_builtin", clock, predicate, enable, args: _*)(sourceInfo)
      }
      new chisel3.assert.Assert()
    }
    def _applyWithSourceLinePrintable(
      cond:    Bool,
      line:    SourceLineInfo,
      message: Option[Printable]
    )(
      implicit sourceInfo: SourceInfo
    ): chisel3.assert.Assert = {
      message.foreach(Printable.checkScope(_))
      val pable = formatFailureMessage("Assertion", line, cond, message)
      emitIfElseFatalIntrinsic(Module.clock, cond, !Module.reset.asBool, pable)
    }
  }

  object assume {
    def _applyWithSourceLinePrintable(
      cond:    Bool,
      line:    SourceLineInfo,
      message: Option[Printable]
    )(
      implicit sourceInfo: SourceInfo
    ): chisel3.assume.Assume = {
      val id = new chisel3.assume.Assume()
      block(layers.Verification.Assume, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
        message.foreach(Printable.checkScope(_))
        when(!Module.reset.asBool) {
          val formattedMsg = formatFailureMessage("Assumption", line, cond, message)
          Builder.pushCommand(Verification(id, Formal.Assume, sourceInfo, Module.clock.ref, cond.ref, formattedMsg))
        }
      }
      id
    }
  }

  object cover {
    def _applyWithSourceLine(
      cond:    Bool,
      line:    SourceLineInfo,
      message: Option[String]
    )(
      implicit sourceInfo: SourceInfo
    ): chisel3.cover.Cover = {
      val id = new chisel3.cover.Cover()
      block(layers.Verification.Cover, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
        when(!Module.reset.asBool) {
          Builder.pushCommand(Verification(id, Formal.Cover, sourceInfo, Module.clock.ref, cond.ref, ""))
        }
      }
      id
    }
  }

}
