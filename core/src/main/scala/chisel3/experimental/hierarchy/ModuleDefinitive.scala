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

private[chisel3] final class ChiselDefinitive[P] private[chisel3] (
  val providedDerivation: Option[DefinitiveDerivation])
    extends DefinitiveProtoProxy[P] {
  derivation = providedDerivation

  val parent = internal.Builder.currentModule
  parent.map(_.definitives += this.toWrapper)

  override def toString = (isResolved, hasDerivation) match {
    case (true, _)      => s"$proto"
    case (false, false) => "{}"
    case (false, true)  => derivation.get.toString
  }
}
