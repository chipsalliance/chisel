// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

/** Provides a macro that returns the source information at the invocation point.
  */
private[chisel3] object SourceInfoMacro {
  def generate_source_info(c: Context): c.Tree = {
    import c.universe._
    val p = c.enclosingPosition

    val userDir = sys.props.get("user.dir") // Figure out what to do if not provided
    val projectRoot = sys.props.get("chisel.project.root")
    val root = projectRoot.orElse(userDir)

    val path = root.map(r => p.source.file.canonicalPath.stripPrefix(r)).getOrElse(p.source.file.name)
    val pathNoStartingSlash = if (path(0) == '/') path.tail else path

    q"_root_.chisel3.experimental.SourceLine($pathNoStartingSlash, ${p.line}, ${p.column})"
  }
}
