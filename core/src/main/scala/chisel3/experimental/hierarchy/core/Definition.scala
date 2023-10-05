// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.language.experimental.macros
import chisel3._

import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.{DefinitionTransform, DefinitionWrapTransform}
import chisel3.experimental.{BaseModule, SourceInfo}
import firrtl.annotations.{IsModule, ModuleTarget, NoTargetAnnotation}

import scala.annotation.nowarn

/** User-facing Definition type.
  * Represents a definition of an object of type `A` which are marked as @instantiable
  * Can be created using Definition.apply method.
  *
  * These definitions are then used to create multiple [[Instance]]s.
  *
  * @param underlying The internal representation of the definition, which may be either be directly the object, or a clone of an object
  */
final case class Definition[+A] private[chisel3] (private[chisel3] val underlying: Underlying[A])
    extends IsLookupable
    with SealedHierarchy[A] {

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[public]]
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
  def _lookup[B, C](
    that: A => B
  )(
    implicit lookup: Lookupable[B],
    macroGenerated:  chisel3.internal.MacroGenerated
  ): lookup.C = {
    lookup.definitionLookup(that, this)
  }

  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = proto match {
    case value: BaseModule =>
      val newChild = Module.do_pseudo_apply(new experimental.hierarchy.DefinitionClone(value))(
        chisel3.experimental.UnlocatableSourceInfo
      )
      newChild._circuit = value._circuit.orElse(Some(value))
      newChild._parent = None
      Some(newChild)
    case value: IsInstantiable => None
  }

  override def toDefinition: Definition[A] = this
  override def toInstance:   Instance[A] = new Instance(underlying)

}

/** Factory methods for constructing [[Definition]]s */
object Definition extends SourceInfoDoc {
  implicit class DefinitionBaseModuleExtensions[T <: BaseModule](d: Definition[T]) {

    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget: ModuleTarget = d.proto.toTarget

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget: IsModule = d.proto.toAbsoluteTarget
  }

  /** A construction method to build a Definition of a Module
    *
    * @param proto the Module being defined
    *
    * @return the input module as a Definition
    */
  def apply[T <: BaseModule with IsInstantiable](proto: => T): Definition[T] = macro DefinitionTransform.apply[T]

  /** A construction method to build a Definition of a Module
    *
    * @param bc the Module being defined
    *
    * @return the input module as a Definition
    */
  def do_apply[T <: BaseModule with IsInstantiable](
    proto: => T
  )(
    implicit sourceInfo: SourceInfo
  ): Definition[T] = {
    val dynamicContext = {
      val context = Builder.captureContext()
      new DynamicContext(Nil, context.throwOnFirstError, context.warningFilters, context.sourceRoots)
    }
    Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
    dynamicContext.inDefinition = true
    val (ir, module) = Builder.build(Module(proto), dynamicContext, false)
    Builder.components ++= ir.components
    Builder.annotations ++= ir.annotations: @nowarn // this will go away when firrtl is merged
    module._circuit = Builder.currentModule
    dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
    new Definition(Proto(module))
  }

}

/** Stores a [[Definition]] that is imported so that its Instances can be
  * compiled separately.
  */
case class ImportDefinitionAnnotation[T <: BaseModule with IsInstantiable](
  definition:      Definition[T],
  overrideDefName: Option[String] = None)
    extends NoTargetAnnotation
