// See LICENSE for license details.

package chisel3.util

import chisel3._
import chisel3.core.ChiselAnnotation
import firrtl.transforms.{BlackBoxSourceHelper}

trait HasBlackBoxResource extends BlackBox {
  self: BlackBox =>

  def setResource(blackBoxResource: String): Unit = {
    annotate(ChiselAnnotation(self, classOf[BlackBoxSourceHelper], s"resource\n$blackBoxResource"))
  }
}

trait HasBlackBoxInline extends BlackBox {
  self: BlackBox =>

  def setInline(blackBoxName: String, blackBoxInline: String): Unit = {
    annotate(ChiselAnnotation(
      self, classOf[BlackBoxSourceHelper], s"inline\n$blackBoxName\n$blackBoxInline"))
  }
}
