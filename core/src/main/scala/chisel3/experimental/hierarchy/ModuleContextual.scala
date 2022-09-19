// SPDX-License-Identifier: Apache-2.0

package chisel3.experimental.hierarchy
import chisel3.experimental.BaseModule
import chisel3.experimental.hierarchy.core._
import chisel3.internal.sourceinfo.SourceInfo
import chisel3.internal.PseudoModule
import chisel3.internal.firrtl._
import chisel3._
import firrtl.annotations.Named
import firrtl.annotations.IsMember

private[chisel3] final class ChiselContextual[P] private[chisel3] (
  val constructionDerivation: Option[ContextualDerivation],
  val parent:                 Option[BaseModule],
  val sourceInfo: SourceInfo)
    extends ContextualProtoProxy[P] {

  derivation = constructionDerivation
  def debug = sourceInfo.makeMessage(x => s"$x $parentDebug/Contextual($identity, $name)")

  parent.map(_.asInstanceOf[BaseModule].contextuals += this.toWrapper)
}

private[chisel3] final class ChiselMockContextual[P] private[chisel3] (
  val constructionDerivation:  Option[ContextualDerivation],
  override val suffixProxyOpt: Option[ContextualProxy[P]],
  override val parentOpt:      Option[Proxy[BaseModule]])
    extends ContextualMockProxy[P] {

  derivation = constructionDerivation
  val suffixProxy: ContextualProxy[P] = suffixProxyOpt.get
  name = suffixProxy.name
  def debug = sourceInfo.makeMessage(x => s"$x $parentDebug/Contextual($identity, $name)")
  val sourceInfo = suffixProxy.sourceInfo
}
