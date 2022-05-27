// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.Queue
import chisel3.internal.ChiselException

class ToTargetSpec extends ChiselFlatSpec with Utils {

  var m: InstanceNameModule = _
  ChiselStage.elaborate { m = new InstanceNameModule; m }

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

  it should "work with non-hardware values (but be deprecated)" in {
    val (ylog, y) = grabLog(m.y.toTarget.toString)
    val (zlog, z) = grabLog(m.z.toTarget.toString)
    assert(y == s"$top>y")
    ylog should include(deprecationMsg)
    assert(z == s"$top>z")
    zlog should include(deprecationMsg)
  }

  it should "work with non-hardware bundle elements (but be deprecated)" in {
    val (log, foo) = grabLog(m.z.foo.toTarget.toString)
    log should include(deprecationMsg)
    assert(foo == s"$top>z.foo")
  }

  it should "work with modules" in {
    val q = m.q.toTarget.toString
    assert(q == s"~$mn|Queue")
  }
}
