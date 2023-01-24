// SPDX-License-Identifier: Apache-2.0

/** This has to be in a separate file and package to generate the correct class
  *  in the annotation.  This is supper annoying.
  */

package circt

import firrtl.annotations.SingleTargetAnnotation
import firrtl.annotations.ModuleTarget

case class Intrinsic(target: ModuleTarget, intrinsic: String) extends SingleTargetAnnotation[ModuleTarget] {
  def duplicate(newTarget: ModuleTarget) = this.copy(target = newTarget)
}
