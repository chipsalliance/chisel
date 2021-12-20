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
  * @param underlying The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
final case class Instance[+A] /*private [chisel3]*/ (private[chisel3] underlying: Underlying[A]) extends SealedHierarchy[A] {
  underlying match {
    case Proto(p: IsClone[_]) => chisel3.internal.throwException("Cannot have a Proto with a clone!")
    case other => //Ok
  }

  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = underlying match {
    case Proto(value: BaseModule)        => Some(value)
    case Proto(value: IsInstantiable)    => None
    case Clone(i: BaseModule)           => Some(i)
    case Clone(i: InstantiableClone[_]) => i.getInnerContext
  }

  /** @return the context this instance. Note that for non-module clones, getInnerDataContext will be the same as getClonedParent */
  private[chisel3] def getClonedParent: Option[BaseModule] = underlying match {
    case Proto(value: BaseModule) => value._parent
    case Clone(i: BaseModule)           => i._parent
    case Clone(i: InstantiableClone[_]) => i.getInnerContext
  }

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
  override def toDefinition: Definition[A] = new Definition(Proto(proto))
  override def toInstance: Instance[A] = this

}

/** Factory methods for constructing [[Instance]]s */
object Instance extends SourceInfoDoc {
  implicit class InstanceBaseModuleExtensions[T <: BaseModule](i: Instance[T]) {
    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: IsModule = i.underlying match {
      case Proto(x: BaseModule) => x.getTarget
      case Clone(x: IsClone[_] with BaseModule) => x.getTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = i.underlying match {
      case Proto(x) => x.toAbsoluteTarget
      case Clone(x: IsClone[_] with BaseModule) => x.toAbsoluteTarget
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
    new Instance(Clone(clone))
  }

}
