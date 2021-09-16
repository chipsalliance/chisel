// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import scala.reflect.runtime.universe.{WeakTypeTag, TypeTag}

import chisel3._
import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import chisel3.internal.BaseModule.IsClone
import scala.language.existentials

/** User-facing Definition type.
  * Represents a definition of an object of type [[A]] which are marked as @instantiable 
  * Can be created using Definition.apply method.
  * 
  * These definitions are then used to create multiple [[Instance]]s.
  *
  * @param cloned The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
case class Definition[+A] private[chisel3] (private[chisel3] cloned: Either[A, IsClone[A]], private[chisel3] definitionTypeTag: TypeTag[_]) extends IsLookupable with Hierarchy[A] {

  /** Used by Chisel's internal macros. DO NOT USE in your normal Chisel code!!!
    * Instead, mark the field you are accessing with [[@public]]
    *
    * Given a selector function (that) which selects a member from the original, return the
    *   corresponding member from the definition.
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
    lookup.definitionLookup(that, this)
  }

  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = getProto match {
    case value: BaseModule =>
      val newChild = new internal.BaseModule.DefinitionClone(value)
      newChild._circuit = value._circuit.orElse(Some(value))
      newChild._parent = None
      Some(newChild)
    case value: IsInstantiable => None
  }

  def toInstance: Instance[A] = new Instance(Left(getProto), definitionTypeTag)

  def isA[B : WeakTypeTag]: Boolean = definitionTypeTag.tpe <:< implicitly[WeakTypeTag[B]].tpe 
}

/** Factory methods for constructing [[Definition]]s */
object Definition extends SourceInfoDoc {
  implicit class DefinitionBaseModuleExtensions[T <: BaseModule](d: Definition[T]) {
    /** If this is an instance of a Module, returns the toTarget of this instance
      * @return target of this instance
      */
    def toTarget = d.getProto.toTarget

    /** If this is an instance of a Module, returns the toAbsoluteTarget of this instance
      * @return absoluteTarget of this instance
      */
    def toAbsoluteTarget = d.getProto.toAbsoluteTarget
  }

  /** A construction method to build a Definition of a Module
    *
    * @param proto the Module being defined
    *
    * @return the input module as a Definition
    */
  def apply[T <: BaseModule with IsInstantiable : TypeTag](proto: => T) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Definition[T] = {
    val dynamicContext = new DynamicContext(Nil)
    Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
    dynamicContext.inDefinition = true
    val (ir, module) = Builder.build(Module(proto), dynamicContext)
    Builder.components ++= ir.components
    Builder.annotations ++= ir.annotations
    module._circuit = Builder.currentModule
    dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
    new Definition(Left(module), implicitly[TypeTag[T]])
  }
}
