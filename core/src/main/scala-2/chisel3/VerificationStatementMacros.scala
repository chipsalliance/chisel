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
import chisel3.ltl._
import chisel3.ltl.Sequence.BoolSequence

import scala.language.experimental.macros
import scala.reflect.macros.blackbox
import scala.reflect.macros.blackbox.Context
import scala.annotation.nowarn

object VerifStmtMacrosCompat {

  type SourceLineInfo = (String, Int)

  private[chisel3] def formatFailureMessage(
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

  private[chisel3] def getLine(c: blackbox.Context): SourceLineInfo = {
    val p = c.enclosingPosition
    (p.source.file.name, p.line): @nowarn // suppress, there's no clear replacement
  }

  private[chisel3] def resetToDisableMigrationChecks(label: String)(implicit sourceInfo: SourceInfo) = {
    val disable = Module.disable.value
    withDisable(Disable.Never) {
      AssertProperty(
        prop = ltl.Property.eventually(!disable),
        label = Some(s"${label}_never_enabled")
      )
      CoverProperty(!disable, s"${label}_enabled")
    }
  }

  object assert {

    /** @group VerifPrintMacros */
    def _applyMacroWithInterpolatorCheck(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree,
      data:       c.Tree*
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      _checkFormatString(c)(message)
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithStringMessage(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree,
      data:       c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)},_root_.scala.Some(_root_.chisel3.Printable.pack($message,..$data)))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithPrintableMessage(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithNoMessage(
      c:          blackbox.Context
    )(cond:       c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
    }

    /** @group VerifPrintMacros */
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
        resetToDisableMigrationChecks("assertion")
        IfElseFatalIntrinsic(id, format, "chisel3_builtin", clock, predicate, enable, format.unpackArgs: _*)(sourceInfo)
      }
      new chisel3.assert.Assert()
    }
  }

  object assume {

    /** @group VerifPrintMacros */
    def _applyMacroWithInterpolatorCheck(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree,
      data:       c.Tree*
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      _checkFormatString(c)(message)
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithStringMessage(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree,
      data:       c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some(_root_.chisel3.Printable.pack($message, ..$data)))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithPrintableMessage(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithNoMessage(
      c:          blackbox.Context
    )(cond:       c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLinePrintable"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
    }

    /** @group VerifPrintMacros */
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
        resetToDisableMigrationChecks("assumption")
        when(!Module.reset.asBool) {
          val formattedMsg = formatFailureMessage("Assumption", line, cond, message)
          Builder.pushCommand(Verification(id, Formal.Assume, sourceInfo, Module.clock.ref, cond.ref, formattedMsg))
        }
      }
      id
    }
  }

  object cover {

    /** @group VerifPrintMacros */
    def _applyMacroWithNoMessage(
      c:          blackbox.Context
    )(cond:       c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.None)($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyMacroWithMessage(
      c:          blackbox.Context
    )(cond:       c.Tree,
      message:    c.Tree
    )(sourceInfo: c.Tree
    ): c.Tree = {
      import c.universe._
      val apply_impl_do = symbolOf[this.type].asClass.module.info.member(TermName("_applyWithSourceLine"))
      q"$apply_impl_do($cond, ${getLine(c)}, _root_.scala.Some($message))($sourceInfo)"
    }

    /** @group VerifPrintMacros */
    def _applyWithSourceLine(
      cond:    Bool,
      line:    SourceLineInfo,
      message: Option[String]
    )(
      implicit sourceInfo: SourceInfo
    ): chisel3.cover.Cover = {
      val id = new chisel3.cover.Cover()
      block(layers.Verification.Cover, skipIfAlreadyInBlock = true, skipIfLayersEnabled = true) {
        resetToDisableMigrationChecks("cover")
        when(!Module.reset.asBool) {
          Builder.pushCommand(Verification(id, Formal.Cover, sourceInfo, Module.clock.ref, cond.ref, ""))
        }
      }
      id
    }
  }
}
