// SPDX-License-Identifier: Apache-2.0

package chisel3

import chisel3.aop.Select
import chisel3.stage.{ChiselGeneratorAnnotation, DesignAnnotation}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SelectSpec extends AnyFlatSpec with Matchers {

  "Placeholders" should "be examined" in {
    class Foo extends RawModule {
      val placeholder = new Placeholder()
      val a = Wire(Bool())
      val b = placeholder.append {
        Wire(Bool())
      }
    }
    val design = ChiselGeneratorAnnotation(() => {
      new Foo
    }).elaborate(1).asInstanceOf[DesignAnnotation[Foo]].design
    Select.wires(design).size should be(2)
  }

}
