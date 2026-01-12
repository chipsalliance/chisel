// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import java.io.File.separatorChar
import java.nio.file.Path

///////////////////////////////////////////////////////
//                     WARNING!!                     //
// This file is soft-linked into the compiler plugin //
// so that the logic stays consistent                //
///////////////////////////////////////////////////////

/** Scala compile-time function for determine the String used to represent a source file path in SourceInfos */
private[internal] object SourceInfoFileResolver {
  // Ensure non-empty paths end in a slash
  def sanitizePath(path: String): String = {
    if (path.isEmpty) path
    else if (path.last != separatorChar) path + separatorChar
    else path
  }

  def resolve(source: Path): String = {
    val userDir = sys.props.get("user.dir") // Figure out what to do if not provided
    val projectRoot = sys.props.get("chisel.project.root")
    val root = projectRoot.orElse(userDir).map(sanitizePath)

    root.map(r => source.toRealPath().toString.stripPrefix(r)).getOrElse(source.toString)
  }
}
