// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.experimental.LazyAnnotation
import firrtl.transforms.{BlackBoxResourceAnno, BlackBoxInlineAnno, BlackBoxSourceHelper}
import firrtl.annotations.RunTransformAnnotation

trait HasBlackBoxResource extends BlackBox {
  self: BlackBox =>

  def setResource(blackBoxResource: String): Unit = {
    val anno = LazyAnnotation(() => BlackBoxResourceAnno(self.toNamed, blackBoxResource))
    chisel3.experimental.annotate(anno)
    chisel3.experimental.annotate(RunTransformAnnotation(classOf[BlackBoxSourceHelper]))
  }
}

trait HasBlackBoxInline extends BlackBox {
  self: BlackBox =>

  def setInline(blackBoxName: String, blackBoxInline: String): Unit = {
    val anno = LazyAnnotation(() => BlackBoxInlineAnno(self.toNamed, blackBoxName, blackBoxInline))
    chisel3.experimental.annotate(anno)
    chisel3.experimental.annotate(RunTransformAnnotation(classOf[BlackBoxSourceHelper]))
  }
}
