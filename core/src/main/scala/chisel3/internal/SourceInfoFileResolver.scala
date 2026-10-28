// SPDX-License-Identifier: Apache-2.0

package chisel3.internal.sourceinfo

import java.io.File.separatorChar
import java.nio.file.Path
import java.util.concurrent.ConcurrentHashMap

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

  def resolve(source: Path, macroSettings: List[String] = Nil): String = {
    val userDir = sys.props.get("user.dir") // Figure out what to do if not provided
    // Later settings override earlier ones, matching compiler flag semantics
    val projectRoot = macroSettings.reverseIterator.collectFirst {
      case setting if setting.startsWith("chisel.project.root=") =>
        setting.stripPrefix("chisel.project.root=")
    }
      .orElse(sys.props.get("chisel.project.root"))
    val root = projectRoot.orElse(userDir).map(sanitizePath)

    root match {
      case None    => source.toString
      case Some(r) => cache.computeIfAbsent((source, r), _ => source.toRealPath().toString.stripPrefix(r))
    }
  }

  // Resolving is called for every SourceInfo macro expansion and Module definition, but there are
  // few distinct source files. toRealPath hits the filesystem so it is worth caching. The root is
  // part of the key so that resident compilers (e.g. sbt, Mill, Bloop) reusing this classloader
  // still respect changes to the project root.
  private val cache = new ConcurrentHashMap[(Path, String), String]
}
