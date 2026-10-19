// SPDX-License-Identifier: Apache-2.0

package chisel3.internal

import scala.annotation.nowarn

/** Builds includers that select auto-tests for elaboration. */
object AutoTestIncluder {

  /** Create an auto-test includer that does not include any tests. */
  @nowarn("cat=deprecation")
  def none: AutoTestIncluder = InlineTestIncluder.none

  /** Create an auto-test includer that includes all tests. */
  @nowarn("cat=deprecation")
  def all: AutoTestIncluder = InlineTestIncluder.all

  /** Create an auto-test includer with module and test name globs. */
  @nowarn("cat=deprecation")
  def apply(
    includeModuleGlobs:   Seq[String],
    includeTestNameGlobs: Seq[String]
  ): AutoTestIncluder = InlineTestIncluder(includeModuleGlobs, includeTestNameGlobs)
}
