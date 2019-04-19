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

class SyncReadMemTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = SyncReadMem(2, UInt(2.W))
  val rdata = mem.read(cnt - 1.U, cnt =/= 0.U)

  switch (cnt) {
    is (0.U) { mem.write(cnt, 3.U) }
    is (1.U) { mem.write(cnt, 2.U) }
    is (2.U) { assert(rdata === 3.U) }
    is (3.U) { assert(rdata === 2.U) }
    is (4.U) { stop() }
  }
}

class SyncReadMemWithZeroWidthTester extends BasicTester {
  val (cnt, _) = Counter(true.B, 3)
  val mem      = SyncReadMem(2, UInt(0.W))
  val rdata    = mem.read(0.U, true.B)

  switch (cnt) {
    is (1.U) { assert(rdata === 0.U) }
    is (2.U) { stop() }
  }
}

class MemorySpec extends ChiselPropSpec {
  property("Mem of Vec should work") {
    assertTesterPasses { new MemVecTester }
  }

  property("SyncReadMem should work") {
    assertTesterPasses { new SyncReadMemTester }
  }

  property("SyncReadMem should work with zero width entry") {
    assertTesterPasses { new SyncReadMemWithZeroWidthTester }
  }
}
