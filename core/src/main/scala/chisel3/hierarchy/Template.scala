// SPDX-License-Identifier: Apache-2.0

package chisel3

import scala.collection.immutable.ListMap
import scala.collection.mutable.{ArrayBuffer, HashMap}
import scala.language.experimental.macros

import java.util.IdentityHashMap

import chisel3.internal._
import chisel3.internal.Builder._
import chisel3.internal.firrtl._
import chisel3.internal.sourceinfo.{InstTransform, SourceInfo}
import chisel3.experimental.BaseModule
import _root_.firrtl.annotations.{ModuleName, ModuleTarget, IsModule}

object Template extends SourceInfoDoc {
  /** A wrapper method that all Module instantiations must be wrapped in
    * (necessary to help Chisel track internal state).
    *
    * @param bc the Module being created
    *
    * @return the input module `m` with Chisel metadata properly set
    */
  def apply[T <: RawModule](bc: => T): Template[T] = macro InstTransform.apply[T]

  /** @group SourceInfoTransformMacro */
  def do_apply[T <: RawModule](bc: => T) (implicit sourceInfo: SourceInfo, compileOptions: CompileOptions): Template[T] = {
    val dynamicContext = new DynamicContext()
    Builder.globalNamespace.copyTo(dynamicContext.globalNamespace)
    val (ir, module) = Builder.build(Module(bc), dynamicContext)
    Builder.components ++= ir.components
    Builder.annotations ++= ir.annotations
    dynamicContext.globalNamespace.copyTo(Builder.globalNamespace)
    new Template(module, "blah")
  }
}

case class Template[T <: BaseModule] private (module: T, other: String)