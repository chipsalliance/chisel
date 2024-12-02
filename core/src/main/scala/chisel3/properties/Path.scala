// SPDX-License-Identifier: Apache-2.0

package chisel3.properties

import chisel3.{Data, HasTarget, MemBase, Module, SramTarget}
import chisel3.experimental.BaseModule
import firrtl.annotations.{InstanceTarget, IsMember, ModuleTarget, ReferenceTarget}
import firrtl.ir.PathPropertyLiteral

/** Represent a Path type for referencing a hardware instance or member in a Property[Path]
  */
sealed abstract class Path {
  private[chisel3] def convert(): PathPropertyLiteral
}

/** Represent a Path type with a known target.
  */
private[properties] sealed abstract class TargetPath extends Path {
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

/** Represent a Path type for a target that no longer exists.
  */
private[properties] object DeletedPath extends Path {
  private[chisel3] def convert(): PathPropertyLiteral = PathPropertyLiteral("OMDeleted:")
}

object Path {

  /** Construct a Path that refers to a Module
    */
  def apply(module: BaseModule): Path = apply(module, false)
  def apply(module: BaseModule, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new TargetPath {
      private val scope = Module.currentModule
      def toTarget():   IsMember = module.toRelativeTarget(scope)
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path that refers to a Data
    */
  def apply(data: Data): Path = apply(data, false)
  def apply(data: Data, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new TargetPath {
      private val scope = Module.currentModule
      def toTarget():   IsMember = data.toRelativeTarget(scope)
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path that refers to a Memory
    */
  def apply(mem: MemBase[_]): Path = apply(mem, false)
  def apply(mem: MemBase[_], isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new TargetPath {
      private val scope = Module.currentModule
      def toTarget():   IsMember = mem.toRelativeTarget(scope)
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path that refers to a SRAMInterface constructed memory
    */
  private[chisel3] def apply(mem: SramTarget): Path = {
    new TargetPath {
      private val scope = Module.currentModule
      def toTarget():   IsMember = mem.toRelativeTarget(scope)
      def isMemberPath: Boolean = false
    }
  }

  /** Construct a Path that refers to a HasTarget
    */
  def apply(hasTarget: HasTarget): Path = apply(hasTarget, false)
  def apply(hasTarget: HasTarget, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new TargetPath {
      private val scope = Module.currentModule
      def toTarget():   IsMember = hasTarget.toRelativeTarget(scope)
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path from a target
    */
  def apply(target: => IsMember): Path = apply(target, false)
  def apply(target: => IsMember, isMemberPath: Boolean): Path = {
    val _isMemberPath = isMemberPath // avoid name shadowing below
    new TargetPath {
      def toTarget():   IsMember = target
      def isMemberPath: Boolean = _isMemberPath
    }
  }

  /** Construct a Path for a target that no longer exists.
    */
  def deleted: Path = DeletedPath
}
