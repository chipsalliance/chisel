// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{Data, MemBase}
import chisel3.experimental.BaseModule
import firrtl.annotations.{InstanceTarget, IsMember, ModuleTarget, ReferenceTarget}
import firrtl.ir.{PathPropertyLiteral}

/** Represent a Path type for referencing a hardware instance or member in a Property[PathType]
  */
sealed abstract class PathType {
  private[chisel3] def toTarget(): IsMember

  private[chisel3] def convert(): PathPropertyLiteral = {
    val target = toTarget()
    val targetType = target match {
      case _: ModuleTarget    => "OMModuleTarget"
      case _: InstanceTarget  => "OMInstanceTarget"
      case _: ReferenceTarget => "OMReferenceTarget"
    }
    PathPropertyLiteral(s"$targetType:${target.serialize}")
  }
}

object PathType {

  /** Construct a PathType that refers to a Module
    */
  def apply(module: BaseModule): PathType = {
    new PathType {
      def toTarget(): IsMember = module.toAbsoluteTarget
    }
  }

  /** Construct a PathType that refers to a Data
    */
  def apply(data: Data): PathType = {
    new PathType {
      def toTarget(): IsMember = data.toAbsoluteTarget
    }
  }

  /** Construct a PathType that refers to a Memory
    */
  def apply(mem: MemBase[_]): PathType = {
    new PathType {
      def toTarget(): IsMember = mem.toAbsoluteTarget
    }
  }
}
