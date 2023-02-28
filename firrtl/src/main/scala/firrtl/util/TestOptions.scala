// SPDX-License-Identifier: Apache-2.0

package firrtl.util

import ClassUtils.isClassLoaded

object TestOptions {
  // Our timing is inaccurate if we're running tests under coverage.
  // If any of the classes known to be associated with evaluating coverage are loaded,
  //  assume we're running tests under coverage.
  // NOTE: We assume we need only ask the class loader that loaded us.
  // If it was loaded by another class loader (outside of our hierarchy), it wouldn't be available to us.
  val coverageClasses = List("scoverage.Platform", "com.intellij.rt.coverage.instrumentation.TouchCounter")
  val accurateTiming = !coverageClasses.exists(isClassLoaded(_))
}
