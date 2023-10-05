// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import chisel3._
import chisel3.testers.BasicTester
import chisel3.util.{Counter, PriorityMux}
import chiselTests.ChiselFlatSpec
import _root_.circt.stage.ChiselStage.emitCHIRRTL

class PriorityMuxTester extends BasicTester {

  val sel = Wire(UInt(3.W))
  sel := 0.U // default

  val elts = Seq(5.U, 6.U, 7.U)
  val muxed = PriorityMux(sel, elts)

  // Priority is given to lowest order bit
  val tests = Seq(
    1.U -> elts(0),
    2.U -> elts(1),
    3.U -> elts(0),
    4.U -> elts(2),
    5.U -> elts(0),
    6.U -> elts(1),
    7.U -> elts(0)
  )
  val (cycle, done) = Counter(0 until tests.size + 1)

  for (((in, out), idx) <- tests.zipWithIndex) {
    when(cycle === idx.U) {
      sel := in
      assert(muxed === out)
    }
  }

  when(done) {
    stop()
  }
}

class PriorityMuxSpec extends ChiselFlatSpec {
  behavior.of("PriorityMux")

  it should "be functionally correct" in {
    assertTesterPasses(new PriorityMuxTester)
  }

  it should "be stack safe" in {
    emitCHIRRTL(new RawModule {
      val n = 1 << 15
      val in = IO(Input(Vec(n, UInt(8.W))))
      val sel = IO(Input(UInt(n.W)))
      val out = IO(Output(UInt(8.W)))
      out := PriorityMux(sel, in)
    })
  }
}
