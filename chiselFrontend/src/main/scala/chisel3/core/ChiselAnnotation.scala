// See LICENSE for license details.

package chisel3.core

import chisel3.internal.InstanceId
import firrtl.Transform
import firrtl.annotations.{Annotation, CircuitName, ComponentName, ModuleName}

/**
  * This is a stand-in for the firrtl.Annotations.Annotation because at the time this annotation
  * is created the component cannot be resolved, into a targetString.  Resolution can only
  * happen after the circuit is elaborated
  * @param component       A chisel thingy to be annotated, could be module, wire, reg, etc.
  * @param transformClass  A fully-qualified class name of the transformation pass
  * @param value           A string value to be used by the transformation pass
  */
case class ChiselAnnotation(component: InstanceId, transformClass: Class[_ <: Transform], value: String) {
  def toFirrtl: Annotation = {
    val circuitName = CircuitName(component.pathName.split("""\.""").head)
    component match {
      case m: BaseModule =>
        Annotation(
          ModuleName(m.name, circuitName), transformClass, value)
      case _ =>
        Annotation(
          ComponentName(
            component.instanceName, ModuleName(component.parentModName, circuitName)), transformClass, value)
    }
  }
}
