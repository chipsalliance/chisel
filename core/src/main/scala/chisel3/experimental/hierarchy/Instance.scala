// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import chisel3._
import chisel3.internal.BaseModule.{ModuleClone, IsClone, InstantiableClone}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import scala.reflect.runtime.universe.{WeakTypeTag, TypeTag}
import scala.language.existentials

/** User-facing Instance type.
  * Represents a unique instance of type [[A]] which are marked as @instantiable 
  * Can be created using Instance.apply method.
  *
  * @param cloned The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
case class Instance[+A] private [chisel3] (private[chisel3] cloned: Either[A, IsClone[A]], private[chisel3] definitionTypeTag: TypeTag[_]) extends Hierarchy[A] {

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
  def toDefinition: Definition[A] = new Definition(Left(getProto), definitionTypeTag)

  def sharesDefinition[A](other: Instance[A]): Boolean = toDefinition == other.toDefinition

  def isA[B : WeakTypeTag]: Boolean = definitionTypeTag.tpe <:< implicitly[WeakTypeTag[B]].tpe
}

/** Factory methods for constructing [[Instance]]s */
object Instance extends SourceInfoDoc {
  implicit class InstanceBaseModuleExtensions[T <: BaseModule](i: Instance[T]) {
    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget = i.cloned match {
      case Left(x: BaseModule) => x.toTarget
      case Right(x: IsClone[_] with BaseModule) => x.toTarget
    }

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget = i.cloned match {
      case Left(x) => x.toAbsoluteTarget
      case Right(x: IsClone[_] with BaseModule) => x.toAbsoluteTarget
    }
  }

  /** A constructs an [[Instance]] from a [[Definition]]
    *
    * @param definition the Module being created
    * @return an instance of the module definition
    */
  def apply[T <: BaseModule with IsInstantiable : TypeTag](definition: Definition[T])(implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Instance[T] = {
    import chisel3.internal.BaseModule.ClonePorts
    import chisel3.internal.PortBinding
    import chisel3.internal.firrtl.DefInvalid
    val proto = definition.getProto
    require(proto.isClosed, "Can't clone a module before module close")
    // Fake Module to serve as the _parent of the cloned ports
    // We make this before clonePorts because we want it to come up first in naming in
    // currentModule
    val cloneParent = Module.typedApply(new ModuleClone(proto), definition.definitionTypeTag)
    require(proto.isClosed, "Can't clone a module before module close")
    require(cloneParent.getOptionRef.isEmpty, "Can't have ref set already!")
    // Fake Module to serve as the _parent of the cloned ports
    // We don't create this inside the ModuleClone because we need the ref to be set by the
    // currentModule (and not clonePorts)
    val clonePorts = new ClonePorts(proto.getModulePorts: _*)
    clonePorts.bind(PortBinding(cloneParent))
    clonePorts.setAllParents(Some(cloneParent))
    cloneParent._portsRecord = clonePorts
    // Normally handled during Module construction but ClonePorts really lives in its parent's parent
    if (!compileOptions.explicitInvalidate) {
      internal.Builder.pushCommand(DefInvalid(sourceInfo, clonePorts.ref))
    }
    if (proto.isInstanceOf[Module]) {
      clonePorts("clock") := Module.clock
      clonePorts("reset") := Module.reset
    }
    val ports = clonePorts
    val clone = ports._parent.get.asInstanceOf[ModuleClone[T]]
    clone._madeFromDefinition = true
    val inst = new Instance(Right(clone), definition.definitionTypeTag)
    inst
  }

}
