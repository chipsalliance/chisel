// See LICENSE for license details.

package firrtl
package annotations

import firrtl.ir.Expression
import AnnotationUtils.{validModuleName, validComponentName, toExp}

/**
 * Named classes associate an annotation with a component in a Firrtl circuit
 */
sealed trait Named {
  def name: String
  def serialize: String
}

final case class CircuitName(name: String) extends Named {
  if(!validModuleName(name)) throw AnnotationException(s"Illegal circuit name: $name")
  def serialize: String = name
}

final case class ModuleName(name: String, circuit: CircuitName) extends Named {
  if(!validModuleName(name)) throw AnnotationException(s"Illegal module name: $name")
  def serialize: String = circuit.serialize + "." + name
}

final case class ComponentName(name: String, module: ModuleName) extends Named {
  if(!validComponentName(name)) throw AnnotationException(s"Illegal component name: $name")
  def expr: Expression = toExp(name)
  def serialize: String = module.serialize + "." + name
}
