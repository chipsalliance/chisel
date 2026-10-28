// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import chisel3.experimental.{SourceInfo, SourceLine}

import scala.quoted.*

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

  /** Reads `-Xmacro-settings` via reflection so users need not enable experimental mode.
    * If `XmacroSettings` is renamed or removed, fall back to Nil, which makes the resolver use the
    * `chisel.project.root` system property instead of failing every SourceInfo expansion.
    */
  private def macroSettings(using Quotes): List[String] = {
    val compilationInfo: AnyRef = quotes.reflect.CompilationInfo
    xmacroSettingsMethod.get(compilationInfo.getClass) match {
      case Some(method) =>
        try method.invoke(compilationInfo).asInstanceOf[List[String]]
        catch { case scala.util.control.NonFatal(_) => Nil }
      case None => Nil
    }
  }

  /** Caches the reflective lookup of `XmacroSettings` (including failure) since this runs on every
    * SourceInfo expansion. Keyed by class to stay correct if a resident compiler loads different
    * compiler versions.
    */
  private val xmacroSettingsMethod = new ClassValue[Option[java.lang.reflect.Method]] {
    override protected def computeValue(cls: Class[?]): Option[java.lang.reflect.Method] =
      try {
        val method = cls.getMethod("XmacroSettings")
        // Mirror reflectiveSelectable which also makes the method accessible
        try method.setAccessible(true)
        catch { case scala.util.control.NonFatal(_) => () }
        if (classOf[List[?]].isAssignableFrom(method.getReturnType)) Some(method) else None
      } catch { case scala.util.control.NonFatal(_) => None }
  }
}
