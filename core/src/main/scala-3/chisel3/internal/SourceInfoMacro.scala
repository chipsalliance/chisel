// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import chisel3.experimental.{SourceInfo, SourceLine}

import scala.quoted.*
import scala.reflect.Selectable.reflectiveSelectable

/** Provides a macro that returns the source information at the invocation point.
  */
private[chisel3] object SourceInfoMacro {
  def generate_source_info(using Quotes): Expr[SourceInfo] = {
    import quotes.reflect.*

    val pos = Position.ofMacroExpansion
    val file = pos.sourceFile
    val path = file.getJPath.fold(file.path)(SourceInfoFileResolver.resolve(_, macroSettings))

    '{ SourceLine(${ Expr(path) }, ${ Expr(pos.startLine + 1) }, ${ Expr(pos.startColumn) }) }
  }

  /** Reads `-Xmacro-settings` via a structural call so users need not enable experimental mode.
    * If `XmacroSettings` is renamed or removed, fall back to Nil, which makes the resolver use the
    * `chisel.project.root` system property instead of failing every SourceInfo expansion.
    */
  private def macroSettings(using Quotes): List[String] = {
    import quotes.reflect.*
    try { CompilationInfo.asInstanceOf[{ def XmacroSettings: List[String] }].XmacroSettings }
    catch { case scala.util.control.NonFatal(_) => Nil }
  }
}
