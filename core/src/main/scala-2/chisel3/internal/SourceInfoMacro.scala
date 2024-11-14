// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

import chisel3.internal.sourceinfo.SourceInfoFileResolver

/** Provides a macro that returns the source information at the invocation point.
  */
@deprecated("Public APIs in chisel3.internal are deprecated", "Chisel 3.6")
private[chisel3] object SourceInfoMacro {
  def generate_source_info(c: Context): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val path = SourceInfoFileResolver.resolve(p.source)

    q"_root_.chisel3.experimental.SourceLine($path, ${p.line}, ${p.column})"
  }
}
