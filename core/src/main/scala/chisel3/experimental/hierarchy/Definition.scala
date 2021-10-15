// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import chisel3._

import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.{DefinitionTransform, DefinitionWrapTransform, SourceInfo}
import chisel3.experimental.BaseModule
import chisel3.internal.BaseModule.IsClone
import firrtl.annotations.{IsModule, ModuleTarget}

/** User-facing Definition type.
  * Represents a definition of an object of type [[A]] which are marked as @instantiable 
  * Can be created using Definition.apply method.
  * 
  * These definitions are then used to create multiple [[Instance]]s.
  *
  * @param cloned The internal representation of the instance, which may be either be directly the object, or a clone of an object
  */
case class Definition[+A] (cloned: Either[A, IsClone[A]]) extends IsLookupable {
  def proto: A = cloned match {
    case Left(value: A) => value
    case Right(i: IsClone[A]) => i._proto
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
    lookup.definitionLookup(that, this)
  }

  /** Updated by calls to [[apply]], to avoid recloning returned Data's */
  private [chisel3] val cache = HashMap[Data, Data]()


  /** @return the context of any Data's return from inside the instance */
  private[chisel3] def getInnerDataContext: Option[BaseModule] = proto match {
    case value: BaseModule =>
      val newChild = Module.do_apply(new internal.BaseModule.DefinitionClone(value))(chisel3.internal.sourceinfo.UnlocatableSourceInfo, chisel3.ExplicitCompileOptions.Strict)
      newChild._circuit = value._circuit.orElse(Some(value))
      newChild._parent = None
      Some(newChild)
    case value: IsInstantiable => None
  }

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
  def do_apply[T <: BaseModule with IsInstantiable](proto: => T) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Definition[T] = {
    val dynamicContext = new DynamicContext(Nil)
    Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
    dynamicContext.inDefinition = true
    val (ir, module) = Builder.build(Module(proto), dynamicContext)
    Builder.components ++= ir.components
    Builder.annotations ++= ir.annotations
    module._circuit = Builder.currentModule
    dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
    new Definition(Left(module))
  }
}
