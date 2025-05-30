// SPDX-License-Identifier: Apache-2.0
package firrtl

import firrtl.annotations.{Named, SingleTargetAnnotation}

/** Firrtl implementation for verilog attributes
  * @param target       target component to tag with attribute
  * @param description  Attribute string to add to target
  */
@deprecated("All APIs in package firrtl are deprecated.", "Chisel 7.0.0")
case class AttributeAnnotation(target: Named, description: String) extends SingleTargetAnnotation[Named] {
  def duplicate(n: Named): AttributeAnnotation = this.copy(target = n, description = description)
}
