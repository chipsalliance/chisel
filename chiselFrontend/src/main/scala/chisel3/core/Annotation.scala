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
  trait Scope
  val Separator = ","

  trait Value

  case class StringValue(value: String) extends Value

  case object AllRefs     extends Scope
  case object JustThisRef extends Scope

  case class Raw(component: SignalId, scope: Scope, value: Value)

  case class Resolved(componentName: String, value: Value) {
    override def toString: String = {
      s"$Separator$componentName$Separator$value"
    }
  }

  def resolve(raw: Raw): Resolved = {
    val componentName = raw.scope match {
      case JustThisRef => s"${raw.component.pathName}"
      case AllRefs     => s"${raw.component.parentModName}.${raw.component.signalName}"
      case  _          => throwException(s"Unknown annotation scope for ${raw}")
    }
    Resolved(componentName, raw.value)
  }
}