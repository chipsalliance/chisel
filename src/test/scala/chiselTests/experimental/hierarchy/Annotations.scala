// SPDX-License-Identifier: Apache-2.0

package chiselTests.experimental.hierarchy

import _root_.firrtl.annotations._
import chisel3.experimental.{annotate, BaseModule, Targetable}
import chisel3.{Data, HasTarget, MemBase}
import chisel3.experimental.hierarchy.{Definition, Hierarchy, Instance}

// These annotations exist purely for testing purposes
private[hierarchy] object Annotations {
  case class MarkAnnotation(target: IsMember, tag: String) extends SingleTargetAnnotation[IsMember] {
    def duplicate(n: IsMember): Annotation = this.copy(target = n)
  }
  def mark(d:                  Data, tag:         String): Unit = annotate(d)(Seq(MarkAnnotation(d.toTarget, tag)))
  def mark[T <: Data](d:       MemBase[T], tag:   String): Unit = annotate(d)(Seq(MarkAnnotation(d.toTarget, tag)))
  def mark(d:                  HasTarget, tag:    String): Unit = annotate(d)(Seq(MarkAnnotation(d.toTarget, tag)))
  def mark[B <: BaseModule](d: Hierarchy[B], tag: String): Unit = annotate(d)(Seq(MarkAnnotation(d.toTarget, tag)))
  def amark(d: Data, tag: String): Unit = annotate(d)(Seq(MarkAnnotation(d.toAbsoluteTarget, tag)))
  def amark[B <: BaseModule](d: Hierarchy[B], tag: String): Unit = annotate(d)(Seq(MarkAnnotation(d.toTarget, tag)))
}
