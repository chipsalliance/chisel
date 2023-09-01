// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{Data, MemBase}
import chisel3.experimental.BaseModule
import firrtl.annotations.{InstanceTarget, IsMember, ModuleTarget, ReferenceTarget}
import firrtl.ir.{PathPropertyLiteral}

/** Represent a Path type for referencing a hardware instance or member in a Property[Path]
  */
sealed abstract class Path {
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

object Path {

  /** Construct a Path that refers to a Module
    */
  def apply(module: BaseModule): Path = {
    new Path {
      def toTarget(): IsMember = module.toAbsoluteTarget
    }
  }

  /** Construct a Path that refers to a Data
    */
  def apply(data: Data): Path = {
    new Path {
      def toTarget(): IsMember = data.toAbsoluteTarget
    }
  }

  /** Construct a Path that refers to a Memory
    */
  def apply(mem: MemBase[_]): Path = {
    new Path {
      def toTarget(): IsMember = mem.toAbsoluteTarget
    }
  }

  /** Construct a Path that refers to a Memory
    */
  def apply(target: IsMember): Path = {
    new Path {
      def toTarget(): IsMember = target
    }
  }
}
