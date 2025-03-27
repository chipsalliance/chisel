// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import java.nio.file.{FileSystems, PathMatcher, Paths}

class InlineTestIncluder private (includeModuleGlobs: Seq[String], includeTestNameGlobs: Seq[String]) {
  private def copy(
    includeModuleGlobs:   Seq[String] = this.includeModuleGlobs,
    includeTestNameGlobs: Seq[String] = this.includeTestNameGlobs
  ) = new InlineTestIncluder(includeModuleGlobs, includeTestNameGlobs)

  def includeModule(glob: String) = copy(includeModuleGlobs = includeModuleGlobs ++ Seq(glob))
  def includeTest(glob:   String) = copy(includeTestNameGlobs = includeTestNameGlobs ++ Seq(glob))

  private val filesystem = FileSystems.getDefault()

  private def matchesGlob(glob: String, path: String): Boolean = {
    val matcher = filesystem.getPathMatcher(s"glob:$glob")
    matcher.matches(Paths.get(path))
  }

  def shouldElaborateTest(moduleDesiredName: String, testName: String): Boolean = {
    val (resolvedModuleGlobs, resolvedTestNameGlobs) = (includeModuleGlobs, includeTestNameGlobs) match {
      case x @ (Nil, Nil) => x
      case (Nil, ts)      => (Seq("*"), ts)
      case (ms, Nil)      => (ms, Seq("*"))
      case x              => x
    }

    resolvedModuleGlobs.exists { glob => matchesGlob(glob, moduleDesiredName) } &&
    resolvedTestNameGlobs.exists { glob => matchesGlob(glob, testName) }
  }
}

object InlineTestIncluder {
  def none: InlineTestIncluder = new InlineTestIncluder(Nil, Nil)
}
