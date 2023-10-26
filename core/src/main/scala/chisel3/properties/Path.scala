// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{Data, MemBase}
import chisel3.experimental.BaseModule
import firrtl.annotations.{InstanceTarget, IsMember, ModuleTarget, ReferenceTarget}
import firrtl.ir.{PathPropertyLiteral}

/** Represent a Path type for referencing a hardware instance or member in a Property[Path]
  */
sealed abstract class Path {
  private[chisel3] def toTarget():   IsMember
  private[chisel3] def isMemberPath: Boolean

  private[chisel3] def convert(): PathPropertyLiteral = {
    val target = toTarget()
    val targetType = if (isMemberPath) {
      target match {
        case _: ModuleTarget    => "OMMemberInstanceTarget"
        case _: InstanceTarget  => "OMMemberInstanceTarget"
        case _: ReferenceTarget => "OMMemberReferenceTarget"
      }
    } else {
      target match {
        case _: ModuleTarget    => "OMInstanceTarget"
        case _: InstanceTarget  => "OMInstanceTarget"
        case _: ReferenceTarget => "OMReferenceTarget"
      }
    }
    PathPropertyLiteral(s"$targetType:${target.serialize}")
  }
}

object Path {

  /** Construct a Path that refers to a Module
    */
  def apply(module: BaseModule): Path = apply(module, false)
  def apply(module: BaseModule, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new Path {
      def toTarget():   IsMember = module.toAbsoluteTarget
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path that refers to a Data
    */
  def apply(data: Data): Path = apply(data, false)
  def apply(data: Data, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new Path {
      def toTarget():   IsMember = data.toAbsoluteTarget
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path that refers to a Memory
    */
  def apply(mem: MemBase[_]): Path = apply(mem, false)
  def apply(mem: MemBase[_], isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new Path {
      def toTarget():   IsMember = mem.toAbsoluteTarget
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path from a target
    */
  def apply(target: IsMember): Path = apply(target, false)
  def apply(target: IsMember, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new Path {
      def toTarget():   IsMember = target
      def isMemberPath: Boolean = _isMemberPath
    }
  }
}
