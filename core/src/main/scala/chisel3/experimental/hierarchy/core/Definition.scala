// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy.core

import scala.collection.mutable.HashMap
import scala.reflect.runtime.universe.TypeTag
import scala.language.experimental.macros
import chisel3.internal.sourceinfo.{DefinitionTransform, InstanceTransform}
import java.util.IdentityHashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.{DefinitionTransform, DefinitionWrapTransform, SourceInfo}
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.Definition
import firrtl.annotations.{IsModule, ModuleTarget, NoTargetAnnotation}

/** Represents a Definition of a proto, at the root of a hierarchical path
  *
  * @param proxy underlying representation of proto with correct internal state
  */
final case class Definition[+P] private[chisel3] (private[chisel3] proxy: DefinitionProxy[P]) extends Root[P] {
  override def toDefinition = this
  private[chisel3] def toResolvedDefinition = {
    proxy.isResolved = true
    new ResolvedDefinition(proxy)
  }
}

object Definition {
  def apply[P](proto: => P): Definition[P] =
    macro DefinitionTransform.apply[P]
  def do_apply[P](proto: => P)(implicit extensions: HierarchicalExtensions[P, _]): Definition[P] = {
    (new Definition(extensions.buildDefinition(proto)))
  }

  ///** A construction method to build a Definition of a Module
  //  *
  //  * @param proto the Module being defined
  //  *
  //  * @return the input module as a Definition
  //  */
  //def apply[T <: BaseModule with IsInstantiable](proto: => T): Definition[T] = macro DefinitionTransform.apply[T]

  ///** A construction method to build a Definition of a Module
  //  *
  //  * @param bc the Module being defined
  //  *
  //  * @return the input module as a Definition
  //  */
  //def do_apply[T <: BaseModule with IsInstantiable](
  //  proto: => T
  //)(
  //  implicit sourceInfo: SourceInfo,
  //  compileOptions:      CompileOptions
  //): Definition[T] = {
  //  val dynamicContext = {
  //    val context = Builder.captureContext()
  //    new DynamicContext(Nil, context.throwOnFirstError, context.warningsAsErrors)
  //  }
  //  Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
  //  dynamicContext.inDefinition = true
  //  val (ir, module) = Builder.build(Module(proto), dynamicContext, false)
  //  Builder.components ++= ir.components
  //  Builder.annotations ++= ir.annotations
  //  module._circuit = Builder.currentModule
  //  dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
  //  new Definition(Proto(module))
  //}

}

/** Stores a [[Definition]] that is imported so that its Instances can be
  * compiled separately.
  */
case class ImportDefinitionAnnotation[T <: BaseModule with IsInstantiable](
  definition:      Definition[T],
  overrideDefName: Option[String] = None)
    extends NoTargetAnnotation
