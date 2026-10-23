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
    // XmacroSettings is experimental in Scala 3. Access it structurally so SourceInfo
    // materialization does not require users to enable experimental features.
    val compilationInfo = CompilationInfo.asInstanceOf[{ def XmacroSettings: List[String] }]
    val path = file.getJPath.fold(file.path)(SourceInfoFileResolver.resolve(_, compilationInfo.XmacroSettings))

    '{ SourceLine(${ Expr(path) }, ${ Expr(pos.startLine + 1) }, ${ Expr(pos.startColumn) }) }
  }
}
