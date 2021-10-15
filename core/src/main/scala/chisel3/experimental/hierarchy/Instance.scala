// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros
import chisel3._
import chisel3.internal.BaseModule.{InstantiableClone, IsClone, ModuleClone}
import chisel3.internal.sourceinfo.{InstanceTransform, SourceInfo}
import chisel3.experimental.BaseModule
import firrtl.annotations.IsModule

/** User-facing Instance type.
  * Represents a unique instance of type [[A]] which are marked as @instantiable 
  * Can be created using Instance.apply method.
  *
  * @param cloned The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
case class Instance[+A] (cloned: Either[A, IsClone[A]]) {

  /** Returns the original object which is instantiated here.
    * If this is an instance of a clone, return that clone's original proto
    *
    * @return the original object which was instantiated
    */
  def proto: A = cloned match {
    case Left(value: A) => value
    case Right(i: IsClone[A]) => i._proto
  }

  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = cloned match {
    case Left(value: BaseModule)        => Some(value)
    case Left(value: IsInstantiable)    => None
    case Right(i: BaseModule)           => Some(i)
    case Right(i: InstantiableClone[_]) => i._parent
  }

  /** @return the context this instance. Note that for non-module clones, getInnerDataContext will be the same as getClonedParent */
  private[chisel3] def getClonedParent: Option[BaseModule] = cloned match {
    case Left(value: BaseModule) => value._parent
    case Right(i: BaseModule)           => i._parent
    case Right(i: InstantiableClone[_]) => i._parent
  }

  /** Updated by calls to [[apply]], to avoid recloning returned Data's */
  private [chisel3] val cache = HashMap[Data, Data]()

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[@public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the instance.
    *
    * Our @instantiable and @public macros generate the calls to this apply method
    *
    * By calling this function, we summon the proper Lookupable typeclass from our implicit scope.
    *
    * @param that a user-specified lookup function
    * @param lookup typeclass which contains the correct lookup function, based on the types of A and B
    * @param macroGenerated a value created in the macro, to make it harder for users to use this API
    */
  def _lookup[B, C](that: A => B)(implicit lookup: Lookupable[B], macroGenerated: chisel3.internal.MacroGenerated): lookup.C = {
    lookup.instanceLookup(that, this)
  }

  /** Returns the definition of this Instance */
  def toDefinition: Definition[A] = new Definition(Left(proto))

}

/** Factory methods for constructing [[Instance]]s */
object Instance extends SourceInfoDoc {
  implicit class InstanceBaseModuleExtensions[T <: BaseModule](i: Instance[T]) {
    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: IsModule = i.cloned match {
      case Left(x: BaseModule) => x.getTarget
      case Right(x: IsClone[_] with BaseModule) => x.getTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.cloned match {
      case Left(x) => x.toAbsoluteTarget
      case Right(x: IsClone[_] with BaseModule) => x.toAbsoluteTarget
    }

  }
  /** A constructs an [[Instance]] from a [[Definition]]
    *
    * @param definition the Module being created
    * @return an instance of the module definition
    */
  def apply[T <: BaseModule with IsInstantiable](definition: Definition[T]): Instance[T] = macro InstanceTransform.apply[T]

  /** A constructs an [[Instance]] from a [[Definition]]
    *
    * @param definition the Module being created
    * @return an instance of the module definition
    */
  def do_apply[T <: BaseModule with IsInstantiable](definition: Definition[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    val ports = experimental.CloneModuleAsRecord(definition.proto)
    val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
    clone._madeFromDefinition = true
    new Instance(Right(clone))
  }

}
