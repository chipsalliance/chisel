// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import _root_.firrtl.annotations._
import chisel3.experimental.{annotate, BaseModule}
import chisel3.Data
import chisel3.experimental.hierarchy.{Instance, Definition, Hierarchy}

object Annotations {
  case class MarkAnnotation(target: IsMember, tag: String) extends SingleTargetAnnotation[IsMember] {
    def duplicate(n: IsMember): Annotation = this.copy(target = n)
  }
  case class MarkChiselHierarchyAnnotation[B <: BaseModule](d: Hierarchy[B], tag: String, isAbsolute: Boolean) extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = MarkAnnotation(d.toTarget, tag)
  }
  case class MarkChiselAnnotation(d: Data, tag: String, isAbsolute: Boolean) extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = if(isAbsolute) MarkAnnotation(d.toAbsoluteTarget, tag) else MarkAnnotation(d.toTarget, tag)
  }
  def mark(d: Data, tag: String): Unit = annotate(MarkChiselAnnotation(d, tag, false))
  def mark[B <: BaseModule](d: Hierarchy[B], tag: String): Unit = annotate(MarkChiselHierarchyAnnotation(d, tag, true))
  def amark(d: Data, tag: String): Unit = annotate(MarkChiselAnnotation(d, tag, true))
  def amark[B <: BaseModule](d: Hierarchy[B], tag: String): Unit = annotate(MarkChiselHierarchyAnnotation(d, tag, true))
}
