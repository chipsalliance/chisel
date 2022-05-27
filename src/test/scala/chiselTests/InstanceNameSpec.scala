// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.stage.ChiselStage
import chisel3.util.Queue

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

class InstanceNameSpec extends ChiselFlatSpec with Utils {
  behavior.of("instanceName")
  val moduleName = "InstanceNameModule"
  var m: InstanceNameModule = _
  ChiselStage.elaborate { m = new InstanceNameModule; m }

  val deprecationMsg = "Accessing the .instanceName or .toTarget of non-hardware Data is deprecated"

  it should "work with module IO" in {
    val io = m.io.pathName
    assert(io == moduleName + ".io")
  }

  it should "work for literals" in {
    val x = m.x.pathName
    assert(x == moduleName + ".UInt<2>(\"h03\")")
  }

  it should "work with non-hardware values (but be deprecated)" in {
    val (ylog, y) = grabLog(m.y.pathName)
    val (zlog, z) = grabLog(m.z.pathName)
    ylog should include(deprecationMsg)
    assert(y == moduleName + ".y")
    zlog should include(deprecationMsg)
    assert(z == moduleName + ".z")
  }

  it should "work with non-hardware bundle elements (but be deprecated)" in {
    val (log, foo) = grabLog(m.z.foo.pathName)
    log should include(deprecationMsg)
    assert(foo == moduleName + ".z.foo")
  }

  it should "work with modules" in {
    val q = m.q.pathName
    assert(q == moduleName + ".q")
  }
}
