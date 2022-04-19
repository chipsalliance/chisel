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

/** Proxy of Instance when viewed from a different Hierarchy
  *
  * Represents a non-local instance.
  *
  * @param suffixProxy Proxy of the same proto with a less-specific hierarchical path
  * @param contexts contains contextual values when viewed from this proxy
  */
private[chisel3] final class ModuleDefinitive[P] private[chisel3] (
    val predecessorOption: Option[(DefinitiveProxy[_], Any => P)]
) extends DefinitiveProxy[P] {

  val parent = internal.Builder.currentModule
  parent.map(_.definitives += this.toWrapper)
}