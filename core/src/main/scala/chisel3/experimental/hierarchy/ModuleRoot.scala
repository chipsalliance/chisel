// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy

import scala.language.experimental.macros
import chisel3._
import chisel3.internal.sourceinfo.SourceInfo

import chisel3.internal.{HasId, PseudoModule}
import chisel3.internal.firrtl._
import chisel3.experimental.hierarchy.core._
import scala.collection.mutable.HashMap
import chisel3.internal.{Builder, DynamicContext}
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.experimental.BaseModule
import firrtl.annotations.{IsModule, ModuleTarget}

/** Proxy of Definition of a user-defined module.
  *
  * @param proto Underlying module which this is the definition of
  */
private[chisel3] trait ModuleRoot[T <: BaseModule] extends PseudoModule with RootProxy[T] {
  private[chisel3] var transparentProxy: Option[ModuleTransparent[T]] = None
  def debug = getTarget.toString
}
