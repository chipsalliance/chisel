// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.{DataMirror, FixedPoint}
import chisel3.testers.BasicTester

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

  val q = Module(new util.Queue(UInt(32.W), 4))

  io.bar := io.foo + x
}

class InstanceNameSpec extends ChiselFlatSpec {
  behavior of "instanceName"

  var m: InstanceNameModule = _
  elaborate { m = new InstanceNameModule; m }

  it should "work with module IO" in {
    println(m.io.pathName)
  }

  it should "work with internal vals" in {
    println(m.x.pathName)
    println(m.y.pathName)
    println(m.z.pathName)
  }

  it should "work with bundle elements" in {
    println(m.z.foo.pathName)
  }

  it should "work with modules" in {
    println(m.q.pathName)
  }
}
