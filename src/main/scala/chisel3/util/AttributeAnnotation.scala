// SPDX-License-Identifier: Apache-2.0

package chisel3.utils

import firrtl.AttributeAnnotation
import firrtl.annotations.Named

import chisel3.Data
import chisel3.experimental.{annotate, requireIsAnnotatable, ChiselAnnotation}

object annotateAttribute {
  def apply[T <: Data](target: T, annoString: String): Unit = {
    requireIsAnnotatable(target, "target must be annotatable")
    annotate(new ChiselAnnotation {
      def toFirrtl = AttributeAnnotation(target.toNamed, annoString)
    })
  }
}
