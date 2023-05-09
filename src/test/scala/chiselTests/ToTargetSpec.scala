// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import circt.stage.ChiselStage

class ToTargetSpec extends ChiselFlatSpec with Utils {

  var m: InstanceNameModule = _
  ChiselStage.emitCHIRRTL { m = new InstanceNameModule; m }

  val mn = "InstanceNameModule"
  val top = s"~$mn|$mn"

  behavior.of(".toTarget")

  val deprecationMsg = "Accessing the .instanceName or .toTarget of non-hardware Data is deprecated"

  it should "work with module IO" in {
    val io = m.io.toTarget.toString
    assert(io == s"$top>io")
  }

  it should "not work for literals" in {
    a[ChiselException] shouldBe thrownBy {
      m.x.toTarget.toString
    }
  }

  it should "NOT work for non-hardware values" in {
    a[ChiselException] shouldBe thrownBy { m.y.toTarget }
    a[ChiselException] shouldBe thrownBy { m.z.toTarget }
  }

  it should "NOT work for non-hardware bundle elements" in {
    a[ChiselException] shouldBe thrownBy { m.z.foo.toTarget }
  }

  it should "work with modules" in {
    val q = m.q.toTarget.toString
    assert(q == s"~$mn|Queue4_UInt32")
  }

  it should "error on non-hardware types and provide information" in {
    class Example extends Module {
      val tpe = UInt(8.W)

      val in = IO(Input(tpe))
      val out = IO(Output(tpe))
      out := in
    }

    val e = the[ChiselException] thrownBy extractCause[ChiselException] {
      var e: Example = null
      circt.stage.ChiselStage.emitCHIRRTL { e = new Example; e }
      e.tpe.toTarget
    }
    e.getMessage should include(
      "You cannot access the .instanceName or .toTarget of non-hardware Data: 'tpe', in module 'Example'"
    )
  }
}
