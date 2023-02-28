// See LICENSE for license details.

package chisel3.util.experimental

import chisel3.experimental.{annotate, ChiselAnnotation}
import firrtl.{FirrtlUserException, RenameMap}
import firrtl.annotations.{Annotation, IsMember, ReferenceTarget, SingleTargetAnnotation}

import scala.collection.mutable

object forceName {

  /** Force the name of this instance to the name its given during Chisel compilation
    *
    * @param instance Instance to name
    */
  def apply(instance: chisel3.experimental.BaseModule, name: String): Unit = {
    annotate(new ChiselAnnotation {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, name)
      }
    })
  }

  /** Force the name of this instance to the name its given during Chisel compilation
    *
    * This will rename after potential renames from other Custom transforms during FIRRTL compilation
    * @param instance Signal to name
    */
  def apply(instance: chisel3.experimental.BaseModule): Unit = {
    annotate(new ChiselAnnotation {
      def toFirrtl = {
        val t = instance.toAbsoluteTarget
        ForceNameAnnotation(t, instance.instanceName)
      }
    })
  }
}

/** Links the user-specified name to force to, with the signal/instance in the FIRRTL design
  *
  * @param target signal/instance to force the name
  * @param name name to force it to be
  */
case class ForceNameAnnotation(target: IsMember, name: String) extends SingleTargetAnnotation[IsMember] {
  def duplicate(n: IsMember): ForceNameAnnotation = this.copy(target = n, name)
}
