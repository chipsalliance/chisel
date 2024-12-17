package chisel3.experimental.hierarchy.core

import chisel3._
import scala.collection.mutable.{HashMap, HashSet}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule

trait Hierarchy[+A] extends HierarchyImpl[A] {

  /** Determine whether underlying proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for underlying proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if underlying proto is of type List[String]
    * @return Whether underlying proto is of provided type (with caveats outlined above)
    */
  def isA[B]: Boolean = false

}

object Hierarchy {
  implicit class HierarchyBaseModuleExtensions[T <: BaseModule](i: Hierarchy[T]) {

    /** Returns the toTarget of this hierarchy
      * @return target of this hierarchy
      */
    def toTarget: IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toTarget
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toTarget
      case _ => throw new InternalErrorException("Match error: toTarget i=$i")
    }

    /** Returns the toAbsoluteTarget of this hierarchy
      * @return absoluteTarget of this Hierarchy
      */
    def toAbsoluteTarget: IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toAbsoluteTarget
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toAbsoluteTarget
      case _ => throw new InternalErrorException("Match error: toAbsoluteTarget i=$i")
    }

    /** Returns the toRelativeTarget of this hierarchy
      * @return relativeTarget of this Hierarchy
      */
    def toRelativeTarget(root: Option[BaseModule]): IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toRelativeTarget(root)
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toRelativeTarget(root)
      case _ => throw new InternalErrorException("Match error: toAbsoluteTarget i=$i")
    }

    /** Returns the toRelativeTarget of this hierarchy
      * @return relativeTarget of this Hierarchy
      */
    def toRelativeTargetToHierarchy(root: Option[Hierarchy[BaseModule]]): IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toRelativeTargetToHierarchy(root)
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toRelativeTargetToHierarchy(root)
      case _ => throw new InternalErrorException("Match error: toAbsoluteTarget i=$i")
    }

  }
}
