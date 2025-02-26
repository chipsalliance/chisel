// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.util.Queue
import circt.stage.ChiselStage
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InstanceNameModule extends Module {
  val io = IO(new Bundle {
    val foo = Input(UInt(32.W))
    val bar = Output(UInt(32.W))
  })
  val x = 3.U
  val y = UInt(8.W)
  val z = new Bundle {
    val foo = UInt(8.W)
  }

  val q = Module(new Queue(UInt(32.W), 4))

  io.bar := io.foo + x
}

class InstanceNameSpec extends AnyFlatSpec with Matchers {
  behavior.of("instanceName")
  val moduleName = "InstanceNameModule"
  var m: InstanceNameModule = _
  ChiselStage.emitCHIRRTL { m = new InstanceNameModule; m }

  it should "work with module IO" in {
    val io = m.io.pathName
    assert(io == moduleName + ".io")
  }

  it should "work for literals" in {
    val x = m.x.pathName
    assert(x == moduleName + ".UInt<2>(0h3)")
  }

  it should "NOT work for non-hardware values" in {
    a[ChiselException] shouldBe thrownBy { m.y.pathName }
    a[ChiselException] shouldBe thrownBy { m.z.pathName }
  }

  it should "NOT work for non-hardware bundle elements" in {
    a[ChiselException] shouldBe thrownBy { m.z.foo.pathName }
  }

  it should "work with modules" in {
    val q = m.q.pathName
    assert(q == moduleName + ".q")
  }
}
