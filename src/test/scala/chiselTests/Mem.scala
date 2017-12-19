// See LICENSE for license details.

package chiselTests

import chisel3._
import chisel3.util._
import chisel3.testers.BasicTester

class MemVecTester extends BasicTester {
  val mem = Mem(2, Vec(2, UInt(8.W)))

  // Circuit style tester is definitely the wrong abstraction here
  val (cnt, wrap) = Counter(true.B, 2)
  mem(0)(0) := 1.U

  when (cnt === 1.U) {
    assert(mem.read(0.U)(0) === 1.U)
    stop()
  }
}

class MemorySpec extends ChiselPropSpec {
  property("Mem of Vec should work") {
    assertTesterPasses { new MemVecTester }
  }
}
