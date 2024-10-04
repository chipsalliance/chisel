// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.experimental.SourceInfo
import chisel3.PrintfMacrosCompat._
import scala.language.experimental.macros
import scala.reflect.macros.blackbox
import scala.reflect.macros.blackbox.Context
import scala.annotation.nowarn

object VerifStmtMacrosCompat {
  type SourceLineInfo = (String, Int)

  def getLine(c: blackbox.Context): SourceLineInfo = {
    val p = c.enclosingPosition
    (p.source.file.name, p.line): @nowarn // suppress, there's no clear replacement
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
  }
}
