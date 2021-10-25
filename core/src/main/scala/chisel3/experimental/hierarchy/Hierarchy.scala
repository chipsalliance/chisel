package chisel3.experimental.hierarchy

import chisel3._
import scala.collection.mutable.{HashMap, HashSet}
import scala.reflect.runtime.universe.WeakTypeTag
import chisel3.internal.BaseModule.IsClone
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule
import scala.annotation.implicitNotFound
import scala.reflect.runtime.universe.TypeTag

trait Hierarchy[+A] {
  private[chisel3] def cloned: Either[A, IsClone[A]]
  private[chisel3] def proto: A = cloned match {
    case Left(value: A) => value
    case Right(i: IsClone[A]) => i._proto
  }
  private[chisel3] def definitionTypeTag: TypeTag[_]
  private[chisel3] def protoTypeString: String = definitionTypeTag.tpe.toString
  // SCALA Reflection API
  def isA[B : WeakTypeTag]: Boolean

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Data, Data]()
  private[chisel3] def getInnerDataContext: Option[BaseModule]


  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[@public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the hierarhcy.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B, C](that: A => B)(implicit lookup: Lookupable[B], macroGenerated: chisel3.internal.MacroGenerated): lookup.C
}

object Hierarchy {
  implicit class HierarchyBaseModuleExtensions[T <: BaseModule](i: Hierarchy[T]) {
    /** Returns the toTarget of this hierarchy
      * @return target of this hierarchy
      */
    def toTarget: IsModule = i match {
      case d: Definition[T] => d.toTarget
      case i: Instance[T]   => i.toTarget
    }

    /** Returns the toAbsoluteTarget of this hierarchy
      * @return absoluteTarget of this Hierarchy
      */
    def toAbsoluteTarget: IsModule = i match {
      case d: Definition[T] => d.toAbsoluteTarget
      case i: Instance[T]   => i.toAbsoluteTarget
    }
  }
}