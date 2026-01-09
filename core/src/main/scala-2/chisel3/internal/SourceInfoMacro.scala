// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

import chisel3.internal.sourceinfo.SourceInfoFileResolver
import scala.reflect.io.AbstractFile

/** Provides a macro that returns the source information at the invocation point.
  */
private[chisel3] object SourceInfoMacro {
  def generate_source_info(c: Context): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition
    val path = {
      val file: AbstractFile = p.source.file
      // No file (null) for things like mdoc and macro-generated code.
      if (file.file == null) {
        file.path
      } else {
        SourceInfoFileResolver.resolve(file.file.toPath)
      }
    }

    q"_root_.chisel3.experimental.SourceLine($path, ${p.line}, ${p.column})"
  }
}
