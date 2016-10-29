// See LICENSE for license details.

package chisel3.core

import chisel3.internal.InstanceId
import firrtl.Annotations.Annotation

/**
  * This is a stand-in for the firrtl.Annotations.Annotation because at the time this annotation
  * is created the component cannot be resolved, into a targetString.  Resolution can only
  * happen after the circuit is elaborated
  * @param transformClass  A fully-qualified class name of the transformation pass
  * @param component       A chisel thingy to be annotated, could be module, wire, reg, etc.
  * @param value           A string value to be used by the transformation pass
  */
case class ChiselAnnotation(transformClass: String, component: InstanceId, value: String) {
  def toFirrtl: Annotation = {
    component match {
      case m: Module => Annotation(transformClass, s"${m.modName}", value)
      case _ => Annotation(transformClass, s"${component.parentModName}.${component.instanceName}", value)
    }
  }
}
