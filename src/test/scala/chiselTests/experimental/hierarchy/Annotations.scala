// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import _root_.firrtl.annotations._
import chisel3.experimental.{annotate, BaseModule}
import chisel3.{Data, MemBase}
import chisel3.experimental.hierarchy.{Definition, Hierarchy, Instance}

// These annotations exist purely for testing purposes
private[hierarchy] object Annotations {
  case class MarkAnnotation(target: IsMember, tag: String) extends SingleTargetAnnotation[IsMember] {
    def duplicate(n: IsMember): Annotation = this.copy(target = n)
  }
  case class MarkChiselHierarchyAnnotation[B <: BaseModule](d: Hierarchy[B], tag: String, isAbsolute: Boolean)
      extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = MarkAnnotation(d.toTarget, tag)
  }
  case class MarkChiselAnnotation(d: Data, tag: String, isAbsolute: Boolean)
      extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = if (isAbsolute) MarkAnnotation(d.toAbsoluteTarget, tag) else MarkAnnotation(d.toTarget, tag)
  }
  case class MarkChiselMemAnnotation[T <: Data](m: MemBase[T], tag: String, isAbsolute: Boolean)
      extends chisel3.experimental.ChiselAnnotation {
    def toFirrtl = if (isAbsolute) MarkAnnotation(m.toAbsoluteTarget, tag) else MarkAnnotation(m.toTarget, tag)
  }
  def mark(d:                   Data, tag:         String): Unit = annotate(MarkChiselAnnotation(d, tag, false))
  def mark[T <: Data](d:        MemBase[T], tag:   String): Unit = annotate(MarkChiselMemAnnotation(d, tag, false))
  def mark[B <: BaseModule](d:  Hierarchy[B], tag: String): Unit = annotate(MarkChiselHierarchyAnnotation(d, tag, true))
  def amark(d:                  Data, tag:         String): Unit = annotate(MarkChiselAnnotation(d, tag, true))
  def amark[B <: BaseModule](d: Hierarchy[B], tag: String): Unit = annotate(MarkChiselHierarchyAnnotation(d, tag, true))
}
