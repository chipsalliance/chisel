// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental

import scala.language.implicitConversions

import chisel3.HasTarget
import chisel3.experimental.hierarchy.Hierarchy
import chisel3.internal.NamedComponent
import firrtl.annotations.IsMember

/** Type class for types that can be converted to a Target
  *
  * See [[AnyTargetable]] for type-erased containers of Targetables
  *
  * @note This uses type classes instead of inheritance because Hierarchy does not constrain its type parameter
  *       and thus not all instances of Hierarchy are Targetable.
  */
sealed trait Targetable[A] {

  /** Returns a FIRRTL IsMember that refers to this object in the elaborated hardware graph */
  def toTarget(a: A): IsMember

  /** Returns a FIRRTL IsMember that refers to the absolute path to this object in the elaborated hardware graph */
  def toAbsoluteTarget(a: A): IsMember

  /** Returns a FIRRTL IsMember that references this object, relative to an optional root.
    *
    * If `root` is defined, the target is a hierarchical path starting from `root`.
    *
    * If `root` is not defined, the target is a hierarchical path equivalent to `toAbsoluteTarget`.
    *
    * @note If `root` is defined, and has not finished elaboration, this must be called within `atModuleBodyEnd`.
    * @note The Targetable must be a descendant of `root`, if it is defined.
    * @note This doesn't have special handling for Views.
    */
  def toRelativeTarget(a: A, root: Option[BaseModule]): IsMember

  /** Returns a FIRRTL IsMember that references this object, relative to an optional root.
    *
    * If `root` is defined, the target is a hierarchical path starting from `root`.
    *
    * If `root` is not defined, the target is a hierarchical path equivalent to `toAbsoluteTarget`.
    *
    * @note If `root` is defined, and has not finished elaboration, this must be called within `atModuleBodyEnd`.
    * @note The Targetable must be a descendant of `root`, if it is defined.
    * @note This doesn't have special handling for Views.
    */
  def toRelativeTargetToHierarchy(a: A, root: Option[Hierarchy[BaseModule]]): IsMember
}

object Targetable {

  // Extension methods for using Targetable as the user expects
  implicit class TargetableSyntax[A](a: A)(implicit targetable: Targetable[A]) {
    def toTarget:         IsMember = targetable.toTarget(a)
    def toAbsoluteTarget: IsMember = targetable.toAbsoluteTarget(a)
    def toRelativeTarget(root:            Option[BaseModule]): IsMember = targetable.toRelativeTarget(a, root)
    def toRelativeTargetToHierarchy(root: Option[Hierarchy[BaseModule]]): IsMember =
      targetable.toRelativeTargetToHierarchy(a, root)
  }

  /** NamedComponent is an awkward private API for all HasId except BaseModule
    *
    * This instance works for [[Data]] and [[MemBase]].
    */
  implicit def forNamedComponent[A <: NamedComponent]: Targetable[A] = new Targetable[A] {
    def toTarget(a:                    A):      IsMember = a.toTarget
    def toAbsoluteTarget(a:            A): IsMember = a.toAbsoluteTarget
    def toRelativeTarget(a:            A, root: Option[BaseModule]): IsMember = a.toRelativeTarget(root)
    def toRelativeTargetToHierarchy(a: A, root: Option[Hierarchy[BaseModule]]): IsMember =
      a.toRelativeTargetToHierarchy(root)
  }

  implicit def forBaseModule[A <: BaseModule]: Targetable[A] = new Targetable[A] {
    def toTarget(a:                    A):      IsMember = a.toTarget
    def toAbsoluteTarget(a:            A): IsMember = a.toAbsoluteTarget
    def toRelativeTarget(a:            A, root: Option[BaseModule]): IsMember = a.toRelativeTarget(root)
    def toRelativeTargetToHierarchy(a: A, root: Option[Hierarchy[BaseModule]]): IsMember =
      a.toRelativeTargetToHierarchy(root)
  }

  implicit def forHierarchy[A <: BaseModule, H[A] <: Hierarchy[A]]: Targetable[H[A]] = new Targetable[H[A]] {
    def toTarget(b:                    H[A]):      IsMember = b.toTarget
    def toAbsoluteTarget(b:            H[A]): IsMember = b.toAbsoluteTarget
    def toRelativeTarget(b:            H[A], root: Option[BaseModule]): IsMember = b.toRelativeTarget(root)
    def toRelativeTargetToHierarchy(b: H[A], root: Option[Hierarchy[BaseModule]]): IsMember =
      b.toRelativeTargetToHierarchy(root)
  }

  implicit def forHasTarget: Targetable[HasTarget] = new Targetable[HasTarget] {
    def toTarget(a:                    HasTarget):      IsMember = a.toTarget
    def toAbsoluteTarget(a:            HasTarget): IsMember = a.toAbsoluteTarget
    def toRelativeTarget(a:            HasTarget, root: Option[BaseModule]): IsMember = a.toRelativeTarget(root)
    def toRelativeTargetToHierarchy(a: HasTarget, root: Option[Hierarchy[BaseModule]]): IsMember =
      a.toRelativeTargetToHierarchy(root)
  }

  implicit def forAnyTargetable: Targetable[AnyTargetable] = new Targetable[AnyTargetable] {
    def toTarget(a:         AnyTargetable):                           IsMember = a.toTarget
    def toAbsoluteTarget(a: AnyTargetable):                           IsMember = a.toAbsoluteTarget
    def toRelativeTarget(a: AnyTargetable, root: Option[BaseModule]): IsMember = a.toRelativeTarget(root)
    def toRelativeTargetToHierarchy(a: AnyTargetable, root: Option[Hierarchy[BaseModule]]): IsMember =
      a.toRelativeTargetToHierarchy(root)
  }
}

/** Existential Type class for types that can be converted to a Target
  *
  * This is useful for containers of Targetables.
  */
sealed trait AnyTargetable {
  type A
  def a:          A
  def targetable: Targetable[A]

  // Convenience methods
  def toTarget:         IsMember = targetable.toTarget(a)
  def toAbsoluteTarget: IsMember = targetable.toAbsoluteTarget(a)
  def toRelativeTarget(root:            Option[BaseModule]): IsMember = targetable.toRelativeTarget(a, root)
  def toRelativeTargetToHierarchy(root: Option[Hierarchy[BaseModule]]): IsMember =
    targetable.toRelativeTargetToHierarchy(a, root)
}

object AnyTargetable {

  /** Convert any Targetable A to an AnyTargetable
    *
    * This effectively erases the type parameter and allows mixing of different concrete Targetable objects.
    */
  def apply[A](a: A)(implicit targetable: Targetable[A]): AnyTargetable = {
    type _A = A
    val _a = a
    val _targetable = targetable
    new AnyTargetable {
      type A = _A
      val a:          A = _a
      val targetable: Targetable[A] = _targetable
    }
  }

  /** Implicit conversion making working with Targetables easier */
  implicit def toAnyTargetable[A: Targetable](a: A): AnyTargetable = apply(a)
}
