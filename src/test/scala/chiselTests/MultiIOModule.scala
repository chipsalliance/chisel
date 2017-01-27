// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.experimental.MultiIOModule
import chisel3.testers.BasicTester

class MultiIOPlusOne extends MultiIOModule {
  val in  = IO(Input(UInt(32.W)))
  val out = IO(Output(UInt(32.W)))

  out := in + 1.asUInt
}

class MultiIOTester extends BasicTester {
  val plusModule = Module(new MultiIOPlusOne)
  plusModule.in := 42.U
  assert(plusModule.out === 43.U)
  stop()
}

class MultiIOSpec extends ChiselFlatSpec {
  "Multiple IOs in MultiIOModule" should "work" in {
    assertTesterPasses({ new MultiIOTester })
  }
}
