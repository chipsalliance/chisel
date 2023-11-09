// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import scala.language.experimental.macros
import scala.reflect.macros.blackbox.Context

///////////////////////////////////////////////////////
//                     WARNING!!                     //
// This file is soft-linked into the compiler plugin //
// so that the logic stays consistent                //
///////////////////////////////////////////////////////

/** Scala compile-time function for determine the String used to represent a source file path in SourceInfos */
private[internal] object SourceInfoFileResolver {
  def resolve(source: scala.reflect.internal.util.SourceFile): String = {
    val userDir = sys.props.get("user.dir") // Figure out what to do if not provided
    val projectRoot = sys.props.get("chisel.project.root")
    val root = projectRoot.orElse(userDir)

    val path = root.map(r => source.file.canonicalPath.stripPrefix(r)).getOrElse(source.file.name)
    val pathNoStartingSlash = if (path(0) == '/') path.tail else path
    pathNoStartingSlash
  }
}
