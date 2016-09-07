// See LICENSE for license details.

package chisel3.core

import chisel3.internal.firrtl.Circuit
import chisel3.internal.{throwException, InstanceId}

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

  object Scope {
    trait ScopeType

    trait Specific extends ScopeType  // Annotation applies only to this specific instance
    trait General extends ScopeType   // Annotation applies to all instances of this component
    trait All extends ScopeType       // Debugging only: name becomes composite of all InstanceId API methods
  }

  def resolveAnnotations(circuit: Circuit): Unit = {
    for (annotation <- circuit.annotations) {
      annotation.resolve
    }
  }
}

abstract class Annotation extends Annotation.Scope.ScopeType {
  def component: InstanceId
  private var _firttlInstanceName : Option[String] = None
  def firrtlInstanceName: String = {
    _firttlInstanceName.getOrElse(resolve)
  }
  def isResolved: Boolean = _firttlInstanceName.isDefined

  def resolve: String = {
    val name = this match {
      case _: Annotation.Scope.Specific => s"${component.pathName}"
      case _: Annotation.Scope.General => s"${component.parentModName}.${component.instanceName}"
      case _: Annotation.Scope.All =>
        f"$component%-29s" +
          f"${component.instanceName}%-25s" +
          f"${component.parentModName}%-25s" +
          f"${component.pathName}%-40s" +
          f"${component.parentPathName}%-35s"
      case _ =>
        throwException(s"Annotation $this has unknown scope")
    }
    _firttlInstanceName = Some(name)
    name
  }
}

/**
  * An example annotation that will resolve component in as a specific reference
 *
  * @param component  // chisel component to be annotated
  * @param value      // a string value to associate with the comnponent
  */
case class StringAnnotation(component: InstanceId, value: String) extends Annotation.Scope.Specific
