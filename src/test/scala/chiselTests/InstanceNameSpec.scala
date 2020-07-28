// See LICENSE for license details.

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

class InstanceNameSpec extends ChiselFlatSpec {
  behavior of "instanceName"
  val moduleName = "InstanceNameModule"
  var m: InstanceNameModule = _
  ChiselStage.elaborate { m = new InstanceNameModule; m }

  it should "work with module IO" in {
    val io = m.io.pathName
    assert(io == moduleName + ".io")
  }

  it should "work with internal vals" in {
    val x = m.x.pathName
    val y = m.y.pathName
    val z = m.z.pathName
    assert(x == moduleName + ".UInt<2>(\"h03\")")
    assert(y == moduleName + ".y")
    assert(z == moduleName + ".z")
  }

  it should "work with bundle elements" in {
    val foo = m.z.foo.pathName
    assert(foo == moduleName + ".z.foo")
  }

  it should "work with modules" in {
    val q = m.q.pathName
    assert(q == moduleName + ".q")
  }
}
