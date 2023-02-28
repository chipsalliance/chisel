// SPDX-License-Identifier: Apache-2.0

package firrtl
package transforms

import firrtl.annotations._

/** A component, e.g. register etc. Must be declared only once under the TopAnnotation */
case class NoDedupAnnotation(target: ModuleTarget) extends SingleTargetAnnotation[ModuleTarget] {
  def duplicate(n: ModuleTarget): NoDedupAnnotation = NoDedupAnnotation(n)
}
