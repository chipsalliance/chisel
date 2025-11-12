// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3._

import scala.collection.mutable.{HashMap, HashSet}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule

import scala.annotation.implicitNotFound

/** Super-trait for Instance and Definition
  *
  * Enables writing functions which are Instance/Definition agnostic
  */
sealed trait Hierarchy[+A] extends HierarchyIsA[A] {

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Data, Data]()
  private[chisel3] def getInnerDataContext: Option[BaseModule]

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the hierarchy.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B, C](
    that: A => B
  )(
    implicit lookup: Lookupable[B],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): lookup.C

  /** @return Return the underlying Definition[A] of this Hierarchy[A] */
  def toDefinition: Definition[A]

  /** @return Convert this Hierarchy[A] as a top-level Instance[A] */
  def toInstance: Instance[A]
}

// Used to effectively seal Hierarchy, without requiring Definition and Instance to be in this file.
private[chisel3] trait SealedHierarchy[+A] extends Hierarchy[A]

object Hierarchy {
  implicit class HierarchyBaseModuleExtensions[T <: BaseModule](i: Hierarchy[T]) {

    /** Returns the toTarget of this hierarchy
      * @return target of this hierarchy
      */
    def toTarget: IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toTarget
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toTarget
      case _ => throw new InternalErrorException(s"Match error: toTarget i=$i")
    }

    /** Returns the toAbsoluteTarget of this hierarchy
      * @return absoluteTarget of this Hierarchy
      */
    def toAbsoluteTarget: IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toAbsoluteTarget
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toAbsoluteTarget
      case _ => throw new InternalErrorException(s"Match error: toAbsoluteTarget i=$i")
    }

    /** Returns the toRelativeTarget of this hierarchy
      * @return relativeTarget of this Hierarchy
      */
    def toRelativeTarget(root: Option[BaseModule]): IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toRelativeTarget(root)
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toRelativeTarget(root)
      case _ => throw new InternalErrorException(s"Match error: toAbsoluteTarget i=$i")
    }

    /** Returns the toRelativeTarget of this hierarchy
      * @return relativeTarget of this Hierarchy
      */
    def toRelativeTargetToHierarchy(root: Option[Hierarchy[BaseModule]]): IsModule = i match {
      case d: Definition[T] => new Definition.DefinitionBaseModuleExtensions(d).toRelativeTargetToHierarchy(root)
      case i: Instance[T]   => new Instance.InstanceBaseModuleExtensions(i).toRelativeTargetToHierarchy(root)
      case _ => throw new InternalErrorException(s"Match error: toAbsoluteTarget i=$i")
    }

    def chirrtlString: String = {
      val component = i.proto._component.get
      ElaboratedCircuit(
        chisel3.internal.firrtl.ir.Circuit(component.name, Seq(component), Nil, firrtl.RenameMap(component.name), Nil, Nil, Nil, false, Nil),
        Nil
      ).serialize(Nil)
    }
  }
}
