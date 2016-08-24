// See LICENSE for license details.

package chisel3.core

import chisel3.internal.{throwException, SignalId}

/**
  * support for annotation of components in chisel circuits, resolves problem with component
  * names not being fully known until circuit elaboration time.
  * Annotations must specify a scope, AllRefs, means all uses of the component across different
  * instantiations of the module will see the same annotation vs. JustThisRef which means the annotation
  * will apply just to the specific instance of the module.  The latter case might be used in situations
  * where the specific parameters of the module may be passed to the firrtl compiler
  *
  * TODO: Serialize annotations using JSON, Problem: Introduces JSON dependency in project
  *
  */
object Annotation {
  val Separator = ","

  trait Value

  trait ScopeType
  trait Absolute extends ScopeType
  trait Relative extends ScopeType
  trait All      extends ScopeType  /* for debugging, key becomes all name api strings */

  // Trivial string annotations included here as an example, with relative and absolute types
  case class AbsoluteStringValue(value: String) extends Value with Absolute
  case class RelativeStringValue(value: String) extends Value with Relative

  case class Raw(component: SignalId, value: Value)

  case class Resolved(componentName: String, value: Value) {
    override def toString: String = {
      s"$componentName$Separator$value"
    }
  }

  def resolve(raw: Raw): Resolved = {
    val componentName = raw.value match {
      case v: Absolute => s"${raw.component.pathName}"
      case v: Relative => s"${raw.component.parentModName}.${raw.component.signalName}"
      case v: All =>
        f"${raw.component}%-29s" +
        f"${raw.component.signalName}%-25s" +
        f"${raw.component.parentModName}%-25s" +
        f"${raw.component.pathName}%-40s" +
        f"${raw.component.parentPathName}%-35s"

      case  _          => throwException(s"Unknown annotation scope for ${raw}")
    }
    Resolved(componentName, raw.value)
  }
}