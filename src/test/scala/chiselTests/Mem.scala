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

class SyncReadMemWriteCollisionTester extends BasicTester
    with chisel3.internal.DisableCommandMemoization { // Work around Firrtl issue #1914
  val (cnt, _) = Counter(true.B, 5)

  // Write-first
  val m0 = SyncReadMem(2, UInt(2.W), SyncReadMem.WriteFirst)
  val rd0 = m0.read(cnt)
  m0.write(cnt, cnt)

  // Read-first
  val m1 = SyncReadMem(2, UInt(2.W), SyncReadMem.ReadFirst)
  val rd1 = m1.read(cnt)
  m1.write(cnt, cnt)

  // Read data from address 0
  when (cnt === 3.U) {
    assert(rd0 === 2.U)
    assert(rd1 === 0.U)
  }

  when (cnt === 4.U) {
    stop()
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

// TODO this can't actually simulate with FIRRTL behavioral mems
class HugeSMemTester(size: BigInt) extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = SyncReadMem(size, UInt(8.W))
  val rdata = mem.read(cnt - 1.U, cnt =/= 0.U)

  switch (cnt) {
    is (0.U) { mem.write(cnt, 3.U) }
    is (1.U) { mem.write(cnt, 2.U) }
    is (2.U) { assert(rdata === 3.U) }
    is (3.U) { assert(rdata === 2.U) }
    is (4.U) { stop() }
  }
}
class HugeCMemTester(size: BigInt) extends BasicTester {
  val (cnt, _) = Counter(true.B, 5)
  val mem = Mem(size, UInt(8.W))
  val rdata = mem.read(cnt)

  switch (cnt) {
    is (0.U) { mem.write(cnt, 3.U) }
    is (1.U) { mem.write(cnt, 2.U) }
    is (2.U) { assert(rdata === 3.U) }
    is (3.U) { assert(rdata === 2.U) }
    is (4.U) { stop() }
  }
}

class MemorySpec extends ChiselPropSpec {
  property("Mem of Vec should work") {
    assertTesterPasses { new MemVecTester }
  }

  property("SyncReadMem should work") {
    assertTesterPasses { new SyncReadMemTester }
  }

  property("SyncReadMem write collision behaviors should work") {
    assertTesterPasses { new SyncReadMemWriteCollisionTester }
  }

  property("SyncReadMem should work with zero width entry") {
    assertTesterPasses { new SyncReadMemWithZeroWidthTester }
  }

  property("Massive memories should be emitted in Verilog") {
    val addrWidth = 65
    val size = BigInt(1) << addrWidth
    val smem = compile(new HugeSMemTester(size))
    smem should include (s"reg /* sparse */ [7:0] mem [0:$addrWidth'd${size-1}];")
    val cmem = compile(new HugeCMemTester(size))
    cmem should include (s"reg /* sparse */ [7:0] mem [0:$addrWidth'd${size-1}];")
  }

  property("Implicit conversions with Mem indices should work") {
    """
    |import chisel3._
    |import chisel3.util.ImplicitConversions._
    |class MyModule extends Module {
    |  val io = IO(new Bundle {})
    |  val mem = Mem(32, UInt(8.W))
    |  mem(0) := 0.U
    |}
    |""".stripMargin should compile
  }
}
