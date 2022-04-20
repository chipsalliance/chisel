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
private[chisel3] final class NonSerializableDefinitive[P] private[chisel3] (
    val pred: Option[(DefinitiveProxy[_], Any => Any)]
) extends NonSerializableDefinitiveProxy[P] {

  predecessorOpt = pred.map(_._1)
  func = pred.map(_._2)

  val parent = internal.Builder.currentModule
  parent.map(_.definitives += this.toWrapper)
}

private[chisel3] final class SerializableDefinitive[P] private[chisel3] (
    val pred: Option[(SerializableDefinitiveProxy[_], DefinitiveFunction[Any, Any])]
) extends SerializableDefinitiveProxy[P] {

  predecessorOpt = pred.map(_._1)
  func = pred.map(_._2)

  val parent = internal.Builder.currentModule
  parent.map(_.definitives += this.toWrapper)

  override def toString = (isEmpty, isSet, valueOpt.isEmpty, func.isEmpty) match {
    case (true, false, _, _)                                 => "{}"
    case (true, true, true, false) if func == Some(Identity) => s"${predecessorOpt.get}"
    case (true, true, true, false)                           => s"${func.get}(${predecessorOpt.get})"
    case (false, _, _, _)                                    => s"$proto"
  }
}