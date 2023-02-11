// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.annotations._

/** Tags an annotation to be consumed by this transform */
case class FlattenAnnotation(target: Named) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named) = FlattenAnnotation(n)
}
