// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import chisel3._

import scala.collection.mutable.{HashMap, HashSet}
import scala.reflect.runtime.universe.TypeTag
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.IsModule

import scala.annotation.implicitNotFound

/** Super-trait for Instance and Definition
  *
  * Enables writing functions which are Instance/Definition agnostic
  */
sealed trait Hierarchy[+A] {
  private[chisel3] def underlying: Underlying[A]
  private[chisel3] def proto: A = underlying match {
    case Proto(value) => value
    case Clone(i: IsClone[A]) => i.getProto
  }

  /** Updated by calls to [[_lookup]], to avoid recloning returned Data's */
  private[chisel3] val cache = HashMap[Data, Data]()
  private[chisel3] def getInnerDataContext: Option[BaseModule]

  /** Determine whether underlying proto is of type provided.
    *
    * @note IMPORTANT: this function requires summoning a TypeTag[B], which will fail if B is an inner class.
    * @note IMPORTANT: this function IGNORES type parameters, akin to normal type erasure.
    * @note IMPORTANT: this function relies on Java reflection for underlying proto, but Scala reflection for provided type
    *
    * E.g. isA[List[Int]] will return true, even if underlying proto is of type List[String]
    * @return Whether underlying proto is of provided type (with caveats outlined above)
    */
  def isA[B: TypeTag]: Boolean = {
    val tptag = implicitly[TypeTag[B]]
    // drop any type information for the comparison, because the proto will not have that information.
    val name = tptag.tpe.toString.takeWhile(_ != '[')
    inBaseClasses(name)
  }

  // This code handles a special-case where, within an mdoc context, the type returned from
  //  scala reflection (typetag) looks different than when returned from java reflection.
  //  This function detects this case and reshapes the string to match.
  private def modifyReplString(clz: String): String = {
    if (clz != null) {
      clz.split('.').toList match {
        case "repl" :: "MdocSession" :: app :: rest => s"$app.this." + rest.mkString(".")
        case other                                  => clz
      }
    } else clz
  }
  private lazy val superClasses = calculateSuperClasses(proto.getClass())
  private def calculateSuperClasses(clz: Class[_]): Set[String] = {
    if (clz != null) {
      Set(modifyReplString(clz.getCanonicalName())) ++
        clz.getInterfaces().flatMap(i => calculateSuperClasses(i)) ++
        calculateSuperClasses(clz.getSuperclass())
    } else {
      Set.empty[String]
    }
  }
  private def inBaseClasses(clz: String): Boolean = superClasses.contains(clz)

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
  }
}
