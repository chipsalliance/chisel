// SPDX-License-Identifier: Apache-2.0

package chiselTests.util

import _root_.circt.stage.ChiselStage.emitCHIRRTL
import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, PriorityMux}
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class PriorityMuxTester extends Module {

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

class PriorityMuxSpec extends AnyFlatSpec with Matchers with ChiselSim {
  behavior.of("PriorityMux")

  it should "be functionally correct" in {
    simulate(new PriorityMuxTester)(RunUntilFinished(9))
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

  it should "give a decent error for empty Seqs" in {
    val e = intercept[IllegalArgumentException] {
      PriorityMux(0.U, Seq.empty[UInt])
    }
    e.getMessage should include("PriorityMux must have a non-empty argument")
  }

  it should "give a error when given different size Seqs" in {
    val e = intercept[ChiselException] {
      emitCHIRRTL(
        new RawModule {
          PriorityMux(Seq(true.B, false.B), Seq(1.U, 2.U, 3.U))
        },
        args = Array("--throw-on-first-error")
      )
    }
    e.getMessage should include("PriorityMuxSpec.scala") // Make sure source locator comes from this file
    e.getMessage should include("PriorityMux: input Seqs must have the same length, got sel 2 and in 3")
  }

  // The input bitvector is sign extended to the width of the sequence
  it should "NOT error when given mismatched selector width and Seq size" in {
    emitCHIRRTL(new RawModule {
      PriorityMux("b10".U(2.W), Seq(1.U, 2.U, 3.U))
    })
  }
}
