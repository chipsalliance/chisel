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
    val userDir = sys.props.get("user.dir")
    val projectRoot = sys.env.get("CHISEL_PROJECT_ROOT")
    val isAbs = sys.env.contains("CHISEL_PROJECT_ROOT_ABSOLUTE")
    val projectRootReplace = sys.env.get("CHISEL_PROJECT_ROOT_REPLACE")
    val root = projectRoot.orElse(userDir)

    val path = root.map(r => source.file.canonicalPath.stripPrefix(r)).getOrElse(source.file.name)
    val pathNoStartingSlash = if (path(0) == '/') path.tail else path

    (root, isAbs, projectRootReplace) match {
      case (None, _, _) =>
        throw new Exception(s"Neither user.dir nor chisel.project.root is not defined.")
      case (Some(_), true, None) =>
        source.file.canonicalPath
      case (Some(_), true, Some(projectRootReplace)) =>
        s"$projectRootReplace/${pathNoStartingSlash}"
      case (Some(_), false, Some(projectRootReplace)) =>
        throw new Exception(
          s"chisel.project.root.replace defined ${projectRootReplace}, but chisel.project.root.absolute is not defined"
        )
      case (Some(_), false, None) =>
        pathNoStartingSlash
    }
  }
}
