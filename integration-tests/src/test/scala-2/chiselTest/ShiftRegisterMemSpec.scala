// SPDX-License-Identifier: Apache-2.0

package chiselTests

import chisel3._
import chisel3.simulator.scalatest.ChiselSim
import chisel3.simulator.stimulus.RunUntilFinished
import chisel3.util.{Counter, ShiftRegister}
import org.scalacheck.{Gen, Shrink}
import org.scalatest.propspec.AnyPropSpec

class ShiftMemTester(n: Int, dp_mem: Boolean) extends Module {
  val (cntVal, done) = Counter(true.B, n)
  val start = 23.U
  val sr = ShiftRegister.mem(cntVal + start, n, true.B, dp_mem, Some("simple_sr"))
  when(RegNext(done)) {
    assert(sr === start)
    stop()
  }
}

class ShiftRegisterMemSpec extends AnyPropSpec with PropertyUtils with ChiselSim {

  implicit val nonNegIntShrinker: Shrink[Int] = Shrink.shrinkIntegral[Int].suchThat(_ >= 0)

  property("ShiftRegister with dual-port SRAM should shift") {
    forAll(Gen.choose(0, 4)) { (shift: Int) =>
      simulate { new ShiftMemTester(shift, true) }(RunUntilFinished(shift + 2))
    }
  }

  property("ShiftRegister with single-port SRAM should shift") {
    forAll(Gen.choose(0, 6).suchThat(_ % 2 == 0)) { (shift: Int) =>
      simulate { new ShiftMemTester(shift, false) }(RunUntilFinished(shift + 2))
    }
  }

  property("ShiftRegister with SRAM should hold its output when disabled") {
    val configs = Seq(0, 1, 2, 3, 4, 5, 8).map((_, true)) ++ Seq(0, 1, 2, 4, 6, 8).map((_, false))
    simulate(new Module {
      val in = IO(Input(UInt(8.W)))
      val en = IO(Input(Bool()))
      val out = IO(Output(Vec(configs.size, UInt(8.W))))
      configs.zipWithIndex.foreach { case ((n, dualPort), i) =>
        out(i) := ShiftRegister.mem(in, n, en, dualPort, None)
      }
    }) { dut =>
      val enables = Seq.fill(3)(false) ++ Seq.fill(16)(true) ++
        Seq.fill(12)(Seq(true, false, true, false, false, true, true, true)).flatten ++
        Seq.fill(8)(false) ++ Seq.fill(16)(true)
      var history = Vector.empty[Int]
      enables.zipWithIndex.foreach { case (en, cycle) =>
        val in = (cycle * 37 + 1) & 0xff
        dut.in.poke(in.U)
        dut.en.poke(en.B)
        configs.zipWithIndex.foreach { case ((n, dualPort), i) =>
          if (history.size >= n) {
            val expected = if (n == 0) in else history(history.size - n)
            dut.out(i).expect(expected.U, s"cycle $cycle, length $n, dual-port $dualPort")
          }
        }
        if (en) history :+= in
        dut.clock.step()
      }
    }
  }
}
