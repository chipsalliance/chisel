// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.annotations._

/** A component that should be preserved
  *
  * DCE treats the component as a top-level sink of the circuit
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class DontTouchAnnotation(target: ReferenceTarget) extends SingleTargetAnnotation[ReferenceTarget] {
  def targets = Seq(target)
  def duplicate(n: ReferenceTarget) = this.copy(n)
}
